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
    plt.axis('off')
    # plt.imsave('/output/object_' + str(count) + '_mask_pred.JPEG', mask_pred)
    plt.cla()
    plt.clf()
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 16))

    ax1.set_title('Input Image')
    ax2.set_title('Token-Semantic Coupled Attention Map')
    ax3.set_title('Binary Map')

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


def rotateImage(image, angle):
    row, col = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def main():
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

    gt_mask_file = '/tscam_images/isolated.JPEG'
    worst_case_file = '/tscam_images/worstcase.JPEG'

    gt_mask_im = Image.open(gt_mask_file)
    worst_case_image = Image.open(worst_case_file)
    gt_mask = get_mask(gt_mask_im, model, transform, device)
    worst_case_mask = get_mask(worst_case_image, model, transform, device)

    # create new image of desired size for padding
    height_1, width_1 = worst_case_mask.shape
    height_2, width_2 = gt_mask.shape

    new_height = max(height_1, height_2)
    new_width = max(width_1, width_2)

    print('height_1', height_1, 'width_1', width_1)
    print('height_2', height_2, 'width_2', width_2)
    print('new_heigth', new_height, 'new_width', new_width)

    gt_mask = mask_padding(gt_mask, new_height, new_width)
    worst_case_mask = mask_padding(worst_case_mask, new_height, new_width)

    print('Padded', 'gt_mask.shape', gt_mask.shape, 'worst_case_mask.shape', worst_case_mask.shape)

    rotation_angle = np.arctan2(x1=width_1, x2=height_1) / np.pi * 180
    print('rotation_angle', rotation_angle)
    rotate_mask = rotateImage(gt_mask, rotation_angle)
    rotate_mask_2 = rotateImage(gt_mask, -rotation_angle)

    print('rotate_mask.shape', rotate_mask.shape)

    overlap = rotate_mask + worst_case_mask
    overlap_2 = rotate_mask_2 + worst_case_mask

    overlap_count = np.sum(np.where(overlap == 2).flatten())
    overlap_count_2 = np.sum(np.where(overlap_2 == 2).flatten())

    print(overlap_count, overlap_count_2)


if __name__ == "__main__":
    main()
