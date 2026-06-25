import os
import gc
import ssl
import yaml
from torch import nn

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from tqdm.auto import tqdm
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import argparse

import lpips

# Diffusers library
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

# Local imports
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.utils import misc
from src.utils.loss import PerceptualLoss, PSNR
from src.utils.datasets import Dataset
import src.utils.misc as misc
import ssl

from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context


def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def main(args):
    # 初始化tensorboard writer
    writer = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, log_with=args.report_to, )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        # 在主进程中初始化tensorboard
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard_logs"))
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "pretrained"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pix2pix = Pix2Pix_Turbo(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, pretrained_path=args.model_url)
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet_modify.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet_modify.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")

    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    psnr = PSNR(255.0).cuda()

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet_modify.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet_modify.conv_in.parameters())
    # 加入 MSF 模块的参数
    layers_to_opt += list(net_pix2pix.unet_modify.MSF.parameters())

    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    dataset_train = Dataset(
        img_size=args.resolution,
        dataset_folder=args.dataset_folder,
        training=True
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )

    dataset_val = Dataset(
        img_size=args.resolution,
        dataset_folder=args.dataset_folder,
        valing=True
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_lpips = accelerator.prepare(net_lpips)

    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networks to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    # net_clip.to(accelerator.device, dtype=weight_dtype)

    l1_loss = nn.L1Loss()
    content_loss = PerceptualLoss()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(
            args.test_dataset_folder,
            model=feat_model, num_workers=0,
            num=None, shuffle=False, seed=0,
            batch_size=8, device=torch.device("cuda"),
            mode="clean", custom_image_tranform=fn_transform,
            description="", verbose=True)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        # 课程学习调度
        train_ds = accelerator.unwrap_model(dl_train).dataset
        # train_ds.set_curriculum_stage(4)

        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                gt_keep_mask = batch["gt_keep_mask"]
                gt = batch["ground_truth"]

                B, C, H, W = gt_keep_mask.shape
                # forward pass
                x_tgt_pred = net_pix2pix(gt_keep_mask)

                # Reconstruction loss
                loss_l1 = l1_loss(x_tgt_pred.float(), gt.float()) * args.lambda_l1
                loss_content, loss_style = content_loss(x_tgt_pred.float(), gt.float())
                loss_lpips = net_lpips(x_tgt_pred.float(), gt.float()).mean() * args.lambda_lpips

                loss_content = loss_content * args.lambda_content
                loss_style = loss_style * args.lambda_style

                psnr_value = psnr(postprocess(gt), postprocess(x_tgt_pred))
                loss = loss_l1 + loss_content + loss_style + loss_lpips

                accelerator.backward(loss, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Generator loss: fool the discriminator
                """
                x_tgt_pred = net_pix2pix(gt_keep_mask)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(gt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l1"] = loss_l1.detach().item()
                    logs["loss_content"] = loss_content.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_style"] = loss_style.detach().item()
                    logs["psnr"] = psnr_value.detach().item()

                    progress_bar.set_postfix(**logs)

                    # 使用tensorboard记录标量值
                    if writer is not None:
                        if global_step % config.tensorboard_freq == 0:  
                            writer.add_scalar('train/lossG', logs["lossG"], global_step)
                            writer.add_scalar('train/lossD', logs["lossD"], global_step)
                            writer.add_scalar('train/loss_l1', logs["loss_l1"], global_step)
                            writer.add_scalar('train/loss_content', logs["loss_content"], global_step)
                            writer.add_scalar('train/loss_lpips', logs["loss_lpips"], global_step)
                            writer.add_scalar('train/loss_style', logs["loss_style"], global_step)
                            writer.add_scalar('train/psnr', logs["psnr"], global_step)

                    # viz some images
                    if global_step % args.viz_freq == 1 and writer is not None:
                        # 将图像转换为网格并添加到tensorboard
                        source_grid = torch.cat([gt_keep_mask[idx].float().detach().cpu() for idx in range(min(B, 4))],dim=1)
                        target_grid = torch.cat([gt[idx].float().detach().cpu() for idx in range(min(B, 4))], dim=1)
                        pred_grid = torch.cat([x_tgt_pred[idx].float().detach().cpu() for idx in range(min(B, 4))], dim=1)

                        # 归一化图像到[0,1]范围
                        # source_grid = (source_grid - source_grid.min()) / (source_grid.max() - source_grid.min())
                        # target_grid = (target_grid - target_grid.min()) / (target_grid.max() - target_grid.min())
                        # pred_grid = (pred_grid - pred_grid.min()) / (pred_grid.max() - pred_grid.min())

                        writer.add_image('train/source', source_grid, global_step)
                        writer.add_image('train/target', target_grid, global_step)
                        writer.add_image('train/model_pred', pred_grid, global_step)

                    # checkpoint the sd-turbo_net
                    if global_step % args.checkpointing_steps == 0 or step % dl_train.total_dataset_length == 0:
                        outf = os.path.join(args.output_dir, "pretrained", f"model{epoch}-{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l1, l_content, l_style, l_psnr, l_lpips = [], [], [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)

                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break

                            gt_keep_mask = batch_val["gt_keep_mask"].cuda()
                            gt = batch_val["ground_truth"].cuda()
                            B, C, H, W = gt_keep_mask.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(gt_keep_mask)
                                # compute the reconstruction losses
                                loss_l1 = l1_loss(x_tgt_pred.float(), gt.float())
                                loss_content, loss_style = content_loss(x_tgt_pred.float(), gt.float())
                                loss_lpips = net_lpips(x_tgt_pred.float(), gt.float()).mean()

                                loss_content = loss_content * args.lambda_content
                                loss_style = loss_style * args.lambda_style

                                psnr_value = psnr(postprocess(gt), postprocess(x_tgt_pred))

                                l_l1.append(loss_l1.item())
                                l_content.append(loss_content.item())
                                l_style.append(loss_style.item())
                                l_psnr.append(psnr_value.item())
                                l_lpips.append(loss_lpips.item())

                            # save outputs images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu())
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)

                        if args.track_val_fid:
                            curr_stats = get_folder_features(
                                os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model,
                                num_workers=0, num=None, shuffle=False, seed=0, batch_size=8,
                                device=torch.device("cuda"),
                                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                            if writer is not None:
                                writer.add_scalar('val/clean_fid', fid_score, global_step)

                        logs["val/l1"] = np.mean(l_l1)
                        logs["val/content"] = np.mean(l_content)
                        logs["val/style"] = np.mean(l_style)
                        logs["val/psnr"] = np.mean(l_psnr)
                        logs["val/lpips"] = np.mean(l_lpips)

                        # 使用tensorboard记录验证指标
                        if writer is not None:
                            writer.add_scalar('val/l1', np.mean(l_l1), global_step)
                            writer.add_scalar('val/content', np.mean(l_content), global_step)
                            writer.add_scalar('val/style', np.mean(l_style), global_step)
                            writer.add_scalar('val/psnr', np.mean(l_psnr), global_step)
                            writer.add_scalar('val/lpips', np.mean(l_lpips), global_step)

                        gc.collect()
                        torch.cuda.empty_cache()

    # 关闭tensorboard writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/acp.yml')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )

    args = parser.parse_args()
    config = misc.get_config(args.config)
    main(config)
