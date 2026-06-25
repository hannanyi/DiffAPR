import os

import numpy as np
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig,get_peft_model
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
p = "src/"
sys.path.append(p)
from .model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, Unet_Modify

def keshihua(tensor, filename):
    # 将张量从 CUDA 转移到 CPU
    tensor = tensor.detach().cpu()

    # 移除批次维度
    tensor = tensor.squeeze(0)  # 现在的形状为 [128, 512, 512]

    # 假设选择第一个、第二个和第三个通道来组成 RGB 图像
    r = tensor[0].numpy()
    g = tensor[1].numpy()
    b = tensor[2].numpy()

    #对每个通道进行归一化处理
    r = (r - r.min()) / (r.max() - r.min()) * 255
    g = (g - g.min()) / (g.max() - g.min()) * 255
    b = (b - b.min()) / (b.max() - b.min()) * 255

    # 将数据类型转换为 uint8
    r = r.astype(np.uint8)
    g = g.astype(np.uint8)
    b = b.astype(np.uint8)

    # 合成为 RGB 图像
    img_rgb = np.stack([r, g, b], axis=-1)
    image = transforms.ToPILImage()(img_rgb)
    image.save("./keshihua/"+filename+'.png')
    # # 使用 Matplotlib 显示 RGB 图像
    # plt.imshow(img_rgb)
    # plt.show()

class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)

class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_path=None, pretrained_model_name_or_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(pretrained_model_name_or_path)

        #mode from stablityai/sd-turbo
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])

            #-----------------------VAE----------------------#
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            self.target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            self.lora_rank_vae = lora_rank_vae

            #-----------------------Unet---------------------#
            unet_modify = Unet_Modify(unet)

            unet_modify = get_peft_model(unet_modify, unet_lora_config, 'lora')
            _sd_unet = unet_modify.state_dict()

            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet_modify.load_state_dict(_sd_unet)

            if "state_dict_fusion" in sd:
                unet_modify.MSF.load_state_dict(sd["state_dict_fusion"])

            self.target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                                        "conv_shortcut", "conv_out","proj_in",
                                        "proj_out", "ff.net.2", "ff.net.0.proj"]
            self.lora_rank_unet = lora_rank_unet


        elif pretrained_path is None:
            print("Initializing sd-turbo_net with random weights")
            #-----------------------VAE---------------------
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            #-----------------------Unet---------------------
            target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                                   "conv_shortcut", "conv_out", "proj_in",
                                   "proj_out", "ff.net.2", "ff.net.0.proj"]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",target_modules=target_modules_unet)
            unet_modify = Unet_Modify(unet)
            unet_modify = get_peft_model(unet_modify, unet_lora_config,'lora')

            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        self.unet_modify, self.vae = unet_modify.to("cuda"), vae.to("cuda")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.vae.decoder.gamma = 1
        self.text_encoder.requires_grad_(False)
        # Timestep = 50
        self.timesteps = torch.tensor([50], device="cuda").long()


    def set_eval(self):
        # Set the network to evaluation mode
        self.unet_modify.eval()
        self.vae.eval()
        self.unet_modify.MSF.eval()  # 设置 MSF 为评估模式

        # Disable gradient computation for the entire unet and vae networks
        self.unet_modify.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_modify.MSF.requires_grad_(False)


    def set_train(self):
        self.unet_modify.train()
        self.vae.train()
        self.unet_modify.MSF.train()  # 设置 MSF 为训练模式

        for n, _p in self.unet_modify.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet_modify.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        # 让 MSF 模块的参数参与训练
        for n, _p in self.unet_modify.MSF.named_parameters():
            _p.requires_grad = True


    def forward(self, gt_keep_mask, prompt=None, prompt_tokens=None):

        if prompt is None:
            prompt = ['Ancient Chinese Painting'] * gt_keep_mask.size(0)
            # prompt = [' '] * gt_keep_mask.size(0)

            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        encoded_control = self.vae.encode(gt_keep_mask).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet_modify(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, )
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        pred_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        pred_image = pred_image.clamp(0, 1)

        return pred_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet_modify.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        # 保存 MultiScaleFusion 的参数
        sd["state_dict_fusion"] = self.unet_modify.MSF.state_dict()

        torch.save(sd, outf)
