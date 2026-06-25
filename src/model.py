import os
import numpy as np
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.utils.peft_utils import scale_lora_layers,unscale_lora_layers

from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self, out_channels=320, BatchNorm=nn.BatchNorm2d):
        super(MultiScaleFusion, self).__init__()

        bn_mom = 0.1
        self.out_channels = out_channels

        self.conv1x1_x1 = nn.Conv2d(1280, out_channels, kernel_size=1, bias=False)
        self.conv1x1_x2 = nn.Conv2d(1280, out_channels, kernel_size=1, bias=False)
        self.conv1x1_x3 = nn.Conv2d(640, out_channels, kernel_size=1, bias=False)
        self.conv1x1_fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
        self.scale0 = nn.Sequential(
            BatchNorm(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, features):
        x1, x2, x3, x4 = features

        y4 = self.scale0(x4)
        x1_resized = F.interpolate(self.conv1x1_x1(x1), size=x4.shape[2:], mode='bilinear', align_corners=False)
        y1 = x1_resized + y4

        x2_resized = F.interpolate(self.conv1x1_x2(x2), size=x4.shape[2:], mode='bilinear', align_corners=False)
        y2 = x2_resized + y4

        x3_resized = self.conv1x1_x3(x3)
        y3 = x3_resized + y4

        fused = torch.cat([y1, y2, y3], dim=1)
        fused = self.conv1x1_fuse(fused)

        output = fused + self.shortcut(x4)

        return output

def make_1step_sched(pretrained_model_name_or_path):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    i = 0
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
        i = i + 1

    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):

    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)

    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")

class Unet_Modify(UNet2DConditionModel):
    def __init__(self, unet):
        super().__init__()
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding

        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.up_blocks = unet.up_blocks
        self.mid_block = unet.mid_block

        #在跳连中增加transformer层
        self.MSF = MultiScaleFusion().to('cuda')

        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out
        # self.mappingnet = MappingNetwork()
        self.noise_level = 0.2

    def forward(self, encoded_control, timesteps, encoder_hidden_states):

        # 1. time
        timesteps = timesteps.expand(encoded_control.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(encoded_control.dtype)
        emb = self.time_embedding(t_emb, None)

        # 2. pre-process
        x = self.conv_in(encoded_control)
        # 3. down
        lora_scale = 1.0
        scale_lora_layers(self, lora_scale)

        down_block_res_samples = (x,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                x, res_samples = downsample_block(hidden_states=x,temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                x, res_samples = downsample_block(hidden_states=x, temb=emb, scale=lora_scale)

            down_block_res_samples += res_samples
        # 4. mid
        x = self.mid_block(x, emb, encoder_hidden_states=encoder_hidden_states,)

        upfeature = []
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
                x = upsample_block( hidden_states=x, temb=emb, res_hidden_states_tuple=res_samples,encoder_hidden_states=encoder_hidden_states,)

            else:
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
                x = upsample_block(hidden_states=x, temb=emb, res_hidden_states_tuple=res_samples, scale=lora_scale,)
            upfeature.append(x)

        #add multiscalefusion
        x = self.MSF(upfeature)
        # 6. post-process
        if self.conv_norm_out:
            x = self.conv_norm_out(x)
            x = self.conv_act(x)

        x = self.conv_out(x)
        unscale_lora_layers(self, lora_scale)

        return x
