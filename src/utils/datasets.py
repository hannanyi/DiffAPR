import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
import cv2
import numpy as np
import random

# from src.utils.newdegraded import apply_degradation

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_size,
        dataset_folder,
        training=False,
        testing=False,
        valing=False
    ):
        super().__init__()

        self.training = training
        self.testing = testing
        self.valing = valing
        self.img_size = img_size

        # 4. 数据路径
        if self.training:
            self.filename = pd.read_csv(os.path.join(dataset_folder, "train_filelist.txt"), header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "train")
            self.dataset_degraded_folder = os.path.join(dataset_folder, "train_degrade")

        elif self.valing:
            self.filename = pd.read_csv( os.path.join(dataset_folder, "val_filelist.txt"), header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "val")
            self.dataset_degraded_folder = "/data1/hn/DataSets/Ancient Chinese Painting/val_degraded"
        elif self.testing:
            self.filename = pd.read_csv(os.path.join(dataset_folder, "val_filelist.txt"),header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "val")
            self.dataset_degraded_folder = os.path.join(dataset_folder, "val_D_now/D/degraded")

        # 6. transform
        self.transform = transforms.Compose([
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(img_size),])

    def __len__(self):
        return len(self.filename)  

    def __getitem__(self, idx):
        filename = self.filename[idx]
        filepath = os.path.join(self.dataset_folder, filename)
        # 1. 读取 GT
        input_img = Image.open(filepath).convert("RGB")
        gt_pil = self.transform(input_img)

        degrade_path = os.path.join(self.dataset_degraded_folder,filename)
        degrade_img = Image.open(degrade_path).convert("RGB")
        gt = F.to_tensor(gt_pil)
        gt_keep_mask = self.transform(degrade_img)
        gt_keep_mask = F.to_tensor(gt_keep_mask)

        return {
            "ground_truth": gt,
            "gt_keep_mask": gt_keep_mask,
            "filename": filename
        }
