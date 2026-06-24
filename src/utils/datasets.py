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
        mold_patch_path,
        mold_color_path,
        shape_db_path,
        training=False,
        testing=False,
        valing=False
    ):
        super().__init__()

        self.training = training
        self.testing = testing
        self.valing = valing
        self.img_size = img_size

        # 1. 读取 shape 数据库（只读取，不重建）
        shape_data = np.load(shape_db_path, allow_pickle=True).item()

        self.shape_db = (
            shape_data["hole"],
            shape_data["crack"],
            shape_data["missing"]
        )

        # 2. 读取霉菌 patch
        self.mold_patches = np.load(mold_patch_path, allow_pickle=True)
        # 3. 读取霉菌颜色
        self.mold_colors = np.load(mold_color_path, allow_pickle=True)

        # 4. 数据路径
        if self.training:
            self.filename = pd.read_csv(os.path.join(dataset_folder, "train_filelist.txt"), header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "train")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "train_allchange_degrade")
            self.dataset_degraded_folder = os.path.join(dataset_folder, "train_now_degrade")


            # self.filename = pd.read_csv(os.path.join(dataset_folder, "real_damaged/src-cut/real_all_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "real_damaged/src-cut/clean_all")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "real_damaged/src-cut/corrupted_all")

        elif self.valing:
            self.filename = pd.read_csv( os.path.join(dataset_folder, "val-50_filelist.txt"), header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "val-50")
            self.dataset_degraded_folder = "/data1/hn/DataSets/Ancient Chinese Painting/val-50_degraded_now"
            # self.filename = pd.read_csv(os.path.join(dataset_folder, "real_damaged/real_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "real_damaged/real-512")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "real_damaged/corrupted-512")
        elif self.testing:
            self.filename = pd.read_csv(os.path.join(dataset_folder, "val_filelist.txt"),header=None)[0].tolist()
            self.dataset_folder = os.path.join(dataset_folder, "val")
            self.dataset_degraded_folder = os.path.join(dataset_folder, "val_D_now/D/degraded")

            # self.filename = pd.read_csv(os.path.join(dataset_folder, "real_damaged/src-cut/real_D_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "real_damaged/src-cut/clean_D_2000")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "real_damaged/src-cut/damaged_2000")
            # #
            # self.filename = pd.read_csv(os.path.join(dataset_folder, "real_damaged/src-cut/real_M_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "real_damaged/src-cut/clean_M_2000")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "real_damaged/src-cut/mold_2000")

            # self.filename = pd.read_csv(os.path.join(dataset_folder, "real_damaged/src-cut/real_F_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "real_damaged/src-cut/clean_F_2000")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "real_damaged/src-cut/fade_2000")
            # self.filename = pd.read_csv(os.path.join(dataset_folder, "train_filelist.txt"), header=None)[0].tolist()
            # self.dataset_folder = os.path.join(dataset_folder, "train")
            # self.dataset_degraded_folder = os.path.join(dataset_folder, "train_now_degrade")

        # 6. transform
        self.transform = transforms.Compose([
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(img_size),])

    def __len__(self):
        return len(self.filename)

    # degradation parameters
    def get_degradation_params(self):
        deg_types = random.choice([
            ["mold"],
            ["damage"],
            ["ink"],
            ["mold", "damage"],
            ["mold", "ink"],
            ["damage", "ink"],
            ["mold", "damage", "ink"]
        ])
        severity = random.randint(1, 3)

        return deg_types, severity

    # main function
    # def __getitem__(self, idx):
    #     filename = self.filename[idx]
    #     filepath = os.path.join(self.dataset_folder, filename)
    #     # 1. 读取 GT
    #     input_img = Image.open(filepath).convert("RGB")
    #     gt_pil = self.transform(input_img)
    #     gt_np = np.array(gt_pil)
    #
    #     # 2. training → 在线退化
    #     if self.training:        # if not self.training:
    #
    #         if random.random() < 0.1:
    #             degrade_np = gt_np.copy()
    #         else:
    #             deg_types, severity = self.get_degradation_params()
    #             # RGB → BGR
    #             gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
    #             degrade_bgr = apply_degradation(gt_bgr.copy(), deg_types, severity,
    #                 self.mold_patches, self.mold_colors, self.shape_db)
    #             # BGR → RGB
    #             degrade_np = cv2.cvtColor(degrade_bgr, cv2.COLOR_BGR2RGB)
    #     # 3. validation / testing
    #     else:
    #         degrade_path = os.path.join(self.dataset_degraded_folder,filename)
    #         degrade_img = Image.open(degrade_path).convert("RGB")
    #         gt = F.to_tensor(gt_pil)
    #         gt_keep_mask = self.transform(degrade_img)
    #         gt_keep_mask = F.to_tensor(gt_keep_mask)
    #
    #         return {
    #             "ground_truth": gt,
    #             "gt_keep_mask": gt_keep_mask,
    #             "filename": filename
    #         }
    #
    #     gt = F.to_tensor(gt_pil)
    #     gt_keep_mask = F.to_tensor(Image.fromarray(degrade_np))
    #
    #     return {
    #         "ground_truth": gt,
    #         "gt_keep_mask": gt_keep_mask,
    #         "filename": filename
    #     }

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
