"""
Main training script for audio tokenizer using BSQ.
Adapted from main_image_tokenizer.py for audio/speech data.
"""

import argparse
from collections import OrderedDict
from omegaconf import OmegaConf
import os
import time
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from timm.utils import ModelEmaV2

from transcoder.data.loader import InfiniteDataLoader
from transcoder.optim.utils import get_params_for_weight_decay
from transcoder.utils import distributed as dist_utils
from transcoder.utils import config as config_utils
from transcoder.utils.meters import AverageMeter, ProgressMeter, SPSMeter
from transcoder.utils.misc import check_loss_nan, get_metrics_dict, get_grad_norm

import wandb


def get_args_parser():
    parser = argparse.ArgumentParser(description='Audio Tokenizer', add_help=False)
    parser.add_argument('config', type=str, help='config')
    parser.add_argument('--output-dir', default='./', help='Output directory')
    parser.add_argument('--resume', default=None, type=str, help='checkpoint to resume')
    parser.add_argument('--eval-freq', default=20_000, type=int, help='evaluation frequency')
    parser.add_argument('--save-freq', default=1_000, type=int, help='save frequency')

    # EMA
    parser.add_argument('--use-ema', action='store_true', help='use exponential moving average')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='decay for EMA')
    parser.add_argument('--cpu-ema', action='store_true', help='put EMA weights on CPU')
    parser.add_argument('--ema-eval-freq', default=40_000, type=int, help='evaluation frequency for EMA')

    # system
    parser.add_argument('--start-iter', default=0, type=int, help='starting iteration')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--vis-freq', default=0, type=int, help='visualization frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--skip-quantize', action='store_true', help='skip quantize at evaluation')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    config = OmegaConf.load(args.config)
    print(config)
    print(args)

    if not args.no_wandb and config.get("wandb", None) is not None and dist_utils.is_main_process():
        run_name = f"{config.wandb.get('run', 'anonymous')}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        try:
            wandb.init(
                project=config.wandb.get("project", "audio-transcoder"),
                name=run_name,
                config=OmegaConf.to_container(config, resolve=True),
                settings=wandb.Settings(code_dir=".")
            )
            with_wandb = True
        except:
            print("Failed to initialize wandb")
            with_wandb = False
    else:
        print("Not using wandb; set wandb in the config to use.")
        with_wandb = False

    config.model.params.clamp_range = config.data.clamp_range
    model = config_utils.instantiate_from_config(config.model)
    model.cuda(args.gpu)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=False, static_graph=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Setup EMA
    if args.use_ema:
        ema = ModelEmaV2(model_without_ddp, decay=args.ema_decay, device='cpu' if args.cpu_ema else None)
        print(f"Using EMA with decay {args.ema_decay}")
    else:
        ema = None

    # Setup loss
    loss_fn = config_utils.instantiate_from_config(config.loss)
    loss_fn.cuda(args.gpu)
    if args.distributed:
        loss_fn = torch.nn.parallel.DistributedDataParallel(
            loss_fn, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=False, static_graph=True)
        loss_fn_without_ddp = loss_fn.module
    else:
        loss_fn_without_ddp = loss_fn

    # Setup optimizer
    params_wd, params_no_wd = get_params_for_weight_decay(model_without_ddp)
    optimizer_config = config.optimizer
    optimizer = config_utils.instantiate_from_config(optimizer_config,
        params=[{'params': params_wd}, {'params': params_no_wd, 'weight_decay': 0.}])
    
    # Setup discriminator optimizer
    params_wd_disc, params_no_wd_disc = get_params_for_weight_decay(loss_fn_without_ddp.discriminator)
    optimizer_disc = config_utils.instantiate_from_config(optimizer_config,
        params=[{'params': params_wd_disc}, {'params': params_no_wd_disc, 'weight_decay': 0.}])

    # Setup LR scheduler
    lr_scheduler = config_utils.instantiate_from_config(
        optimizer_config.lr_scheduler_config,
        optimizer=optimizer,
        max_iter=optimizer_config.max_iter
    )
    
    lr_scheduler_disc = config_utils.instantiate_from_config(
        optimizer_config.lr_scheduler_config,
        optimizer=optimizer_disc,
        max_iter=optimizer_config.max_iter
    )

    # Setup data loaders
    train_dataset = config_utils.instantiate_from_config(config.data.train)
    val_dataset = config_utils.instantiate_from_config(config.data.val)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    train_loader_inf = InfiniteDataLoader(train_loader)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler_disc.load_state_dict(checkpoint['lr_scheduler_disc'])
        loss_fn_without_ddp.load_state_dict(checkpoint['loss_fn_state_dict'])
        args.start_iter = checkpoint['iteration'] + 1
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"Starting from iteration: {args.start_iter}")

    # Mixed precision setup
    use_amp = not optimizer_config.disable_amp
    use_bf16 = optimizer_config.get('use_bf16', False)
    if use_amp:
        scaler = amp.GradScaler(enabled=not use_bf16)
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"Using AMP with dtype: {dtype}")
    else:
        scaler = None
        dtype = torch.float32
        print("Not using AMP")

    if args.evaluate:
        validate(val_loader, model, loss_fn, args, config, 0)
        return

    # Training loop
    print("Starting training...")
    for iteration in range(args.start_iter, optimizer_config.max_iter):
        # Train one iteration
        train_one_iteration(
            train_loader_inf, model, loss_fn, optimizer, optimizer_disc,
            lr_scheduler, lr_scheduler_disc, iteration, args, config,
            scaler, dtype, ema, with_wandb
        )

        # Save checkpoint
        if (iteration + 1) % args.save_freq == 0 and dist_utils.is_main_process():
            save_checkpoint(
                model_without_ddp, optimizer, optimizer_disc,
                lr_scheduler, lr_scheduler_disc, loss_fn_without_ddp,
                iteration, args, ema
            )

        # Validation
        if (iteration + 1) % args.eval_freq == 0:
            validate(val_loader, model, loss_fn, args, config, iteration, with_wandb)

        # EMA validation
        if ema is not None and (iteration + 1) % args.ema_eval_freq == 0:
            validate(val_loader, ema.module, loss_fn, args, config, iteration, with_wandb, prefix='ema_')

    print("Training completed!")


def train_one_iteration(train_loader, model, loss_fn, optimizer, optimizer_disc,
                       lr_scheduler, lr_scheduler_disc, iteration, args, config,
                       scaler, dtype, ema, with_wandb):
    model.train()
    loss_fn.train()

    # Get batch
    audio, _ = next(train_loader)
    audio = audio.cuda(args.gpu, non_blocking=True)

    # Ensure audio has correct shape: (batch, time, freq)
    if audio.dim() == 4:
        audio = audio.squeeze(1)

    # Forward pass with AMP
    with amp.autocast(enabled=(scaler is not None), dtype=dtype):
        reconstructions, quant_loss, quant_info = model(audio)
        
        # Compute loss
        aeloss, log_dict_ae = loss_fn(
            quant_loss, audio, reconstructions, 0, iteration,
            last_layer=model.module.get_last_layer() if args.distributed else model.get_last_layer(),
            split="train"
        )

    # Backward pass for autoencoder
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(aeloss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
    else:
        aeloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Discriminator update
    with amp.autocast(enabled=(scaler is not None), dtype=dtype):
        discloss, log_dict_disc = loss_fn(
            quant_loss, audio, reconstructions, 1, iteration,
            last_layer=model.module.get_last_layer() if args.distributed else model.get_last_layer(),
            split="train"
        )

    optimizer_disc.zero_grad()
    if scaler is not None:
        scaler.scale(discloss).backward()
        scaler.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad_norm_(loss_fn.module.discriminator.parameters() if args.distributed else loss_fn.discriminator.parameters(), 1.0)
        scaler.step(optimizer_disc)
        scaler.update()
    else:
        discloss.backward()
        torch.nn.utils.clip_grad_norm_(loss_fn.module.discriminator.parameters() if args.distributed else loss_fn.discriminator.parameters(), 1.0)
        optimizer_disc.step()

    # Update learning rate
    lr_scheduler.step()
    lr_scheduler_disc.step()

    # Update EMA
    if ema is not None:
        ema.update(model.module if args.distributed else model)

    # Logging
    if iteration % args.print_freq == 0 and dist_utils.is_main_process():
        log_dict = {**log_dict_ae, **log_dict_disc, **quant_info}
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        log_dict['lr_disc'] = optimizer_disc.param_groups[0]['lr']
        
        print(f"Iter [{iteration}/{config.optimizer.max_iter}] " + 
              " ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()]))
        
        if with_wandb:
            wandb.log(log_dict, step=iteration)


def validate(val_loader, model, loss_fn, args, config, iteration, with_wandb=False, prefix=''):
    model.eval()
    loss_fn.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for audio, _ in val_loader:
            audio = audio.cuda(args.gpu, non_blocking=True)
            
            if audio.dim() == 4:
                audio = audio.squeeze(1)

            reconstructions, quant_loss, quant_info = model(audio, skip_quantize=args.skip_quantize)
            
            aeloss, log_dict_ae = loss_fn(
                quant_loss, audio, reconstructions, 0, iteration,
                last_layer=model.module.get_last_layer() if args.distributed and hasattr(model, 'module') else model.get_last_layer(),
                split="val"
            )

            total_loss += aeloss.item()
            total_mse += F.mse_loss(reconstructions, audio).item()
            total_mae += F.l1_loss(reconstructions, audio).item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches

    if dist_utils.is_main_process():
        print(f"Validation [{iteration}] {prefix}Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")
        
        if with_wandb:
            wandb.log({
                f'{prefix}val_loss': avg_loss,
                f'{prefix}val_mse': avg_mse,
                f'{prefix}val_mae': avg_mae,
            }, step=iteration)

    model.train()
    loss_fn.train()


def save_checkpoint(model, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc,
                   loss_fn, iteration, args, ema=None):
    checkpoint = {
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_disc': optimizer_disc.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'lr_scheduler_disc': lr_scheduler_disc.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
    }
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    filename = os.path.join(args.output_dir, f'checkpoint_{iteration:07d}.pth')
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Audio tokenizer training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output directory
    if dist_utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
