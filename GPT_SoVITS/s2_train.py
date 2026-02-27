import warnings
warnings.filterwarnings("ignore")
import os
import logging
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from module import commons
from module.data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from process_ckpt import savee

# Performance tweaks
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

global_step = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global global_step
    hps = utils.get_hparams(stage=2)
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
    
    logger = utils.get_logger(hps.data.exp_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    torch.manual_seed(hps.train.seed)

    # Dataset & Loader
    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.model.version)
    collate_fn = TextAudioSpeakerCollate(version=hps.model.version)
    
    # num_workers=0 is safer for Windows to avoid IO bottlenecks/locking
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=0, 
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    # Models
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, version=hps.model.version).to(device)

    # Optimizer parameters logic
    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad, net_g.parameters())

    optim_g = torch.optim.AdamW(
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {"params": net_g.enc_p.text_embedding.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
            {"params": net_g.enc_p.encoder_text.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
            {"params": net_g.enc_p.mrte.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
        ],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    # Load Checkpoints
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "D_*.pth"), net_d, optim_d)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "G_*.pth"), net_g, optim_g)
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
        if hps.train.pretrained_s2G and os.path.exists(hps.train.pretrained_s2G):
            net_g.load_state_dict(torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"], strict=False)
        if hps.train.pretrained_s2D and os.path.exists(hps.train.pretrained_s2D):
            net_d.load_state_dict(torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"], strict=False)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    print(f"Start training from epoch {epoch_str}")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # Training Step
        net_g.train()
        net_d.train()
        
        for batch_idx, data in enumerate(tqdm(train_loader)):
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
                sv_emb = sv_emb.to(device, non_blocking=True)
            else:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data

            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl, text, text_lengths = ssl.to(device), text.to(device), text_lengths.to(device)
            ssl.requires_grad = False

            with autocast(enabled=hps.train.fp16_run):
                if hps.model.version in {"v2Pro", "v2ProPlus"}:
                    outputs = net_g(ssl, spec, spec_lengths, text, text_lengths, sv_emb)
                else:
                    outputs = net_g(ssl, spec, spec_lengths, text, text_lengths)

                y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl = outputs
                
                mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                y_sliced = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminator Turn
            optim_d.zero_grad()
            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            # Generator Turn
            optim_g.zero_grad()
            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl

            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(f"Train Epoch: {epoch} Step: {global_step} Loss Gen: {loss_gen_all:.4f} Loss Disc: {loss_disc:.4f}")
                utils.summarize(writer=writer, global_step=global_step, scalars={"loss/g/total": loss_gen_all, "loss/d/total": loss_disc, "learning_rate": lr})

            global_step += 1

        # End of Epoch Saving
        if epoch % hps.train.save_every_epoch == 0:
            checkpoint_dir = "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version)
            # Standard checkpoint (massive)
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(checkpoint_dir, f"G_{global_step}.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(checkpoint_dir, f"D_{global_step}.pth"))
            
            # Exportable weights (small)
            if hps.train.if_save_every_weights:
                ckpt = net_g.state_dict()
                savee(ckpt, hps.name + f"_e{epoch}_s{global_step}", epoch, global_step, hps, model_version=hps.model.version)

        scheduler_g.step()
        scheduler_d.step()

if __name__ == "__main__":
    main()