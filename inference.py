import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
from pytorch_fid import fid_score
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.utils.datasets import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--dataset_folder", default="/data1/hn/DataSets/Ancient Chinese Painting", type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", default='./sd-turbo_net')
    parser.add_argument('--model_path', type=str,default='./checkpoint/model.pkl', help='path to a sd-turbo_net state dict to be used')
    parser.add_argument('--output_dir', type=str, default='./output/output_now/D', help='the directory to save the outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    # # initialize the sd-turbo_net
    model = Pix2Pix_Turbo( pretrained_path=args.model_path, pretrained_model_name_or_path=args.pretrained_model_name_or_path,)
    model.set_eval()

    #val dataset
    dataset_val = Dataset(img_size=args.img_size, dataset_folder=args.dataset_folder, testing=True)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    for step, batch_val in enumerate(dl_val):
        gt_keep_mask_batch = batch_val["gt_keep_mask"].cuda()
        gt_batch = batch_val["ground_truth"].cuda()
        filename = batch_val["filename"]

        print(step)
        output_image_batch = model(gt_keep_mask_batch)

        for idx in range(args.batch_size):

            # gt = gt_batch[idx]
            # gt_keep_mask = gt_keep_mask_batch[idx]
            output_image = output_image_batch[idx]

            output_image = transforms.ToPILImage()(output_image.cpu())
            # gt_mask = transforms.ToPILImage()(gt_keep_mask.cpu())
            # gt = transforms.ToPILImage()(gt.cpu())

            # save the outputs image
            # os.makedirs(args.output_dir+'/gt_mask', exist_ok=True)
            # os.makedirs(args.output_dir+'/output', exist_ok=True)
            # os.makedirs(args.output_dir+'/gt', exist_ok=True)

            output_image.save(os.path.join(args.output_dir,filename[idx]))
            # gt_mask.save(os.path.join(args.output_dir,'gt_mask', filename[idx]))
            # gt.save(os.path.join(args.output_dir,'gt', filename[idx]))
