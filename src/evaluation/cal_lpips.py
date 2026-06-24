import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import PIL.Image
import torch.nn.functional as F

import torch
import lpips


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
    image = image.transpose(2, 0, 1) # HWC => CHW
    image = torch.from_numpy(image).float().unsqueeze(0)
    image = image / 127.5 - 1

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    device = torch.device('cuda:0')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    # loss_fn = lpips.LPIPS(net='vgg').to(device)
    loss_fn.eval()

    lpips_l = []
    with torch.no_grad():
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            print(i)
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            name1 = name1.split('.')[0]
            name2 = name2.split('.')[0]
            assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, 'pred_'+name2)

            img1 = read_image(fpath1).to(device)
            img2 = read_image(fpath2).to(device)
            if img1.shape != img2.shape:
                img1 = F.interpolate(img1.to(torch.float32), size=(512,512),mode='bilinear',align_corners=False)
            if img1.shape != img2.shape:
                img2 = F.interpolate(img2.to(torch.float32), size=(512,512),mode='bilinear',align_corners=False)

            lpips_l.append(loss_fn(img1, img2).mean().cpu().numpy())

    res = sum(lpips_l) / len(lpips_l)

    return res


if __name__ == '__main__':
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_D_2000'
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_M_2000'
    # folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/real_damaged/src-cut/clean_F_2000'

    # folder2 = '/data1/hn/code/ACP-main/output_newmold/real_mold'
    # folder2 = '/data1/hn/code/ACP-main/output_newdamaged/real_damaged'
    # folder2 = '/data1/hn/code/ACP-main/output_newfade/real_fade'
    # folder2 = '/data1/hn/code/ACP-main/output_allchange/real_damaged'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/OneStoneNet-main/OneStoneNet-blind/output/real_fade'

    # folder2 = '/data1/hn/code/ACP-main/outputs/real_fade'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Resfusion-main/resfusion_restore_test/results/real_damaged'


    # folder1 = '/data1/hn/code/AttDiff Comparative experiment/APLRL-main/output/ground_truth'
    # folder2 = '/data1/hn/code/ACP-main/outputs/output'
    folder1 = '/data1/hn/DataSets/Ancient Chinese Painting/val'
    # folder1 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/test_p256_thick/gt'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/GRL-Image-Restoration-main/output/real_fade'

    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_damaged'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_mold'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/FocalNet-main/Dehazing/OTS/results/FocalNet/real_fade'


    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Restormer-main/results/F+M'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Restormer-main/results/fade_2000'

    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/APLRL-main/output/generate'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Blind_Omni_Wav_Net-main/outputs/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/OneStoneNet-main/OneStoneNet-blind/output/D'
    # folder2 = '/data1/hn/code/Blind_Omni_Wav_Net-main/results/D'

    # folder2 = '/data1/hn/code/ACP-main/outputs/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/NAFNet-main/outputs/F'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/NAFNet-main/outputs/fade_2000'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/TransCNN-HAE-master/checkpoints/results/D'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/TransCNN-HAE-master/checkpoints/results/real_fade'

    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/ACP/inpainted'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/Resfusion-main/resfusion_restore_test/results/D+F+M'

    folder2 = '/data1/hn/code/AttDiff Comparative experiment/StrDiffusion-main (1)/result/D'
    # folder2 = '/data1/hn/code/AttDiff Comparative experiment/RePaint-main/log/ACP/inpainted'
    # folder2 = '/data1/hn/code/DiffACP Comparative experiment/TransCNN-HAE-master/checkpoints/results/M'
    res = calculate_metrics(folder1, folder2)
    print('lpips: %.4f' % res)
    with open('lpips.txt', 'w') as f:
        f.write('lpips: %.4f' % res)