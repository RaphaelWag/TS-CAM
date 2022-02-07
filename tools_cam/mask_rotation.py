import os
import sys
import datetime
import pprint

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, str_gpus, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
from urllib.request import urlretrieve

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer


def get_mask(im, model, transform, device):
    x = transform(im)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        x_logits, tscams = model(x, True)

    x_probs = F.softmax(x_logits, dim=-1)
    pred_cls_id = x_probs.argmax()  # hardcode this to screw?

    cam_pred = tscams[0, pred_cls_id, :, :].detach().cpu().numpy()
    mask_pred = cv2.resize(cam_pred, im.size)
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask_pred = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)
    # mask_image = (mask_pred[..., np.newaxis] * im).astype("uint8")

    _, mask_pred_binary_map = cv2.threshold(mask_pred,
                                            mask_pred.max() * 0.15, 1,
                                            cv2.THRESH_BINARY)
    return mask_pred_binary_map


def mask_padding(mask, height, width):
    orig_height, orig_width = mask.shape

    # compute center offset
    x_start = (width - orig_width) // 2
    y_start = (height - orig_height) // 2

    # padding for gt mask
    padding = np.zeros(shape=(height, width))

    # copy mask into center of emtpy image
    padding[y_start:y_start + orig_height,
    x_start:x_start + orig_width] = mask

    return padding


def image_padding(image, height, width):
    orig_height, orig_width, channels = image.shape

    # compute center offset
    x_start = (width - orig_width) // 2
    y_start = (height - orig_height) // 2

    # padding for gt mask
    color = (0, 0, 0)
    padding = np.full((height, width, channels), color, dtype=np.uint8)

    # copy mask into center of emtpy image
    padding[y_start:y_start + orig_height,
    x_start:x_start + orig_width] = image

    return padding


def rotateImage(image, angle):
    row, col = image.shape
    center = tuple((np.array([row, col]) + 1) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def get_overlap_mask(gt_mask, usecase_mask, rotation_angle):
    rotate_mask = rotateImage(gt_mask, rotation_angle)
    overlap = rotate_mask * usecase_mask
    overlap_count = np.sum(overlap.flatten())
    return [overlap_count, rotate_mask]


def main():
    count = 0
    config_file = '../configs/ILSVRC/deit_tscam_small_patch16_224.yaml'
    cfg_from_file(config_file)
    cfg.BASIC.ROOT_DIR = '../'

    model = create_deit_model(cfg.MODEL.ARCH, pretrained=False, num_classes=cfg.DATA.NUM_CLASSES, drop_rate=0.0,
                              drop_path_rate=0.1, drop_block_rate=None)
    device = 'cuda'
    model = model.to(device)
    checkpoint = torch.load('/ts-cam-deit-small/ts-cam-deit-small/ILSVRC2012/model_epoch12.pth')
    pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(pretrained_dict)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gt_mask_files = glob('/tscam_images/gt_masks/*.JPEG')
    for gt_mask_file in gt_mask_files:
        screw_type = gt_mask_file.split('.')[0].split('_')[-1]
        usecase_files = glob('/tscam_images/' + screw_type + '/*.JPEG')
        for usecase_file in usecase_files:
            gt_mask_im = Image.open(gt_mask_file)
            usecase_image = Image.open(usecase_file)
            gt_mask = get_mask(gt_mask_im, model, transform, device)
            # crop gt mask
            _, contours, _ = cv2.findContours((gt_mask * 255).astype(np.uint8), cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # normal box
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                bbox = [x, y, x + w, y + h]
            gt_mask = gt_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            usecase_mask = get_mask(usecase_image, model, transform, device)

            # create new image of desired size for padding
            height_1, width_1 = usecase_mask.shape
            height_2, width_2 = gt_mask.shape

            new_size = max(height_1, height_2, width_1, width_2)

            gt_mask = mask_padding(gt_mask, new_size, new_size)
            usecase_mask = mask_padding(usecase_mask, new_size, new_size)
            pad_img = image_padding(np.array(usecase_image), new_size, new_size)
            qudrant_sign = 1 if height_1 > width_1 else -1
            rotation_angle = np.arctan(width_1 / height_1) / np.pi * 180
            rotation_angles = [rotation_angle, -rotation_angle, rotation_angle - qudrant_sign * 10,
                               -rotation_angle + qudrant_sign * 10, rotation_angle - qudrant_sign * 5,
                               -rotation_angle + qudrant_sign * 5, 90 if height_1 < width_1 else 0]

            overlap_masks = [get_overlap_mask(gt_mask, usecase_mask, angle) for angle in rotation_angles]
            max_overlap = np.argmax([e[0] for e in overlap_masks])
            best_mask = overlap_masks[max_overlap][1]
            _, contours, _ = cv2.findContours((best_mask * 255).astype(np.uint8), cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # contour
                c = max(contours, key=cv2.contourArea)
                # rotated box
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
                rot_box_im = cv2.drawContours(pad_img, [box], 0, (36, 255, 12), 3)

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
            ax1.set_title('rotated box')
            ax2.set_title('predicted mask')
            _ = ax1.imshow(rot_box_im)  # Visualize rotated box
            _ = ax2.imshow(usecase_mask)  # Visualize mask
            plt.savefig('/output/object_rotated_box_' + str(count) + '.JPEG')
            count += 1
            plt.cla()
            plt.clf()
            plt.close()

    # TODO resize height of gt to diag of inference image


if __name__ == "__main__":
    main()
