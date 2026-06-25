import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from src.pix2pix_turbo import Pix2Pix_Turbo
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--dataset_folder", default='/data1/hn/code/ACP-main/img/real_damaged', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", default='./sd-turbo_net')
    parser.add_argument('--model_path', type=str,default='./checkpoint/model.pkl')
    parser.add_argument('--output_dir', type=str, default='./output', help='the directory to save the outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    transform = transforms.ToTensor()
    # # initialize the sd-turbo_net
    model = Pix2Pix_Turbo(
                    pretrained_path=args.model_path,
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,)
    model.set_eval()
    captions = ['Ancient Chinese Painting']

    idx = 0

    for filename in os.listdir(args.dataset_folder):
        if filename.endswith(('.png', '.jpg', 'jpeg')):
            gt_keep_mask = Image.open(os.path.join(args.dataset_folder, filename)).convert('RGB')
            gt_keep_mask = transform(gt_keep_mask).unsqueeze(0)

            gt_keep_mask = F.interpolate(gt_keep_mask, size=(512,512), mode='bilinear', align_corners=False)

            output_image = model(gt_keep_mask.cuda())

            output_image = transforms.ToPILImage()(output_image[0].cpu())
            gt_mask = transforms.ToPILImage()(gt_keep_mask[0].cpu())

            # save the outputs image
            os.makedirs(args.output_dir+'/input', exist_ok=True)
            os.makedirs(args.output_dir+'/output', exist_ok=True)

            gt_mask.save(os.path.join(args.output_dir,'input', filename))
            output_image.save(os.path.join(args.output_dir,'output', filename))

            # fusion_image.save(os.path.join(args.output_dir,'fusion', filename[idx]))
            idx = idx + 1


