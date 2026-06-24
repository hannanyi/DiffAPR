import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import PIL.Image
import torch.nn.functional as F
from torchvision import transforms


def build_transform(self, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(img_size),
    ])
    return transform

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_l1(img1, img2):
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    l1 = np.mean(np.abs(img1 - img2))

    return l1


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
    # 添加RGBA转RGB的处理
    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA格式转为RGB
        image = image[:, :, :3]

    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    # image = image.transpose(2, 0, 1) # HWC => CHW

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    psnr_l, ssim_l, dl1_l = [], [], []
    for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
        print(i)
        _, name1 = os.path.split(fpath1)
        _, name2 = os.path.split(fpath2)
        name1 = name1.split('.')[0]
        name2 = name2.split('.')[0]
        assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

        img1 = read_image(fpath1).astype(np.float64)
        img2 = read_image(fpath2).astype(np.float64)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_LINEAR)
        if img1.shape != img2.shape:
            img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)

        psnr_l.append(calculate_psnr(img1, img2))
        ssim_l.append(calculate_ssim(img1, img2))
        dl1_l.append(calculate_l1(img1, img2))

    psnr = sum(psnr_l) / len(psnr_l)
    ssim = sum(ssim_l) / len(ssim_l)
    dl1 = sum(dl1_l) / len(dl1_l)

    return psnr, ssim, dl1


if __name__ == '__main__':
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_D_2000'
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_M_2000'
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_F_2000'

    # folder2 = '/data1/hn/code/ACP-main/output_newmold/real_mold'
    # folder2 = '/data1/hn/code/ACP-main/outputs/real_damaged'
    # folder2 = '/data1/hn/code/ACP-main/output_newdamaged/real_damaged'

    # folder1 = '/data1/hn/DataSets/CelebAMask-HQ/val'
    folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/val'
    # folder1 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/test_p256_thick/gt'

    # folder1 = '/data1/hn/code/AttDiff Comparative experiment/APLRL-main/output/ground_truth'
    # folder2 = '/data1/hn/code/Blind_Omni_Wav_Net-main/results/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/GRL-Image-Restoration-main/output/real_fade'

    # folder2 = '/data1/hn/code/ACP-main/outputs/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_damaged'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_mold'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_fade'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Restormer-main/results/F+M'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Restormer-main/results/fade_2000'

    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/APLRL-main/output/generate'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Blind_Omni_Wav_Net-main/outputs/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/OneStoneNet-main/OneStoneNet-blind/output/D'
    folder2 = '/data1/hn/code/AttDiff Comparative experiment/StrDiffusion-main (1)/result/D'

    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/test_p256_thick/inpainted'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Resfusion-main/resfusion_restore_test/results/D+F+M'
    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/StrDiffusion-main/result/D'
    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/ACP/inpainted'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Resfusion-main/resfusion_restore_test/results/real_damaged'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/NAFNet-main/outputs/F'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/NAFNet-main/outputs/fade_2000'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/TransCNN-HAE-master/checkpoints/results/M'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/TransCNN-HAE-master/checkpoints/results/real_fade'


    psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
    print('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))
    with open('psnr_ssim_l1.txt', 'w') as f:
        f.write('model psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))

