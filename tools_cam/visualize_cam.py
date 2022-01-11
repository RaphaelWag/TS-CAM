import os
import sys
import datetime
import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
        cfg.MODEL.ARCH,
        pretrained=True,
        num_classes=cfg.DATA.NUM_CLASSES,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print(model)
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    print('Preparing networks done!')
    return device, model, cls_criterion


def main():
    config_file = '../configs/ILSVRC/deit_tscam_small_patch16_224.yaml'
    cfg_from_file(config_file)
    cfg.BASIC.ROOT_DIR = '../'

    model = create_deit_model(cfg.MODEL.ARCH, pretrained=False, num_classes=cfg.DATA.NUM_CLASSES, drop_rate=0.0,
                              drop_path_rate=0.1, drop_block_rate=None)
    model = model.cuda()
    checkpoint = torch.load('../ts-cam-deit-small/ts-cam-deit-small/ILSVRC2012/model_epoch12.pth')
    pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(pretrained_dict)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    filename = "../tscam_images/val/object/object_0.JPEG"
    im = Image.open(filename).convert('RGB')
    x = transform(im)

    with torch.no_grad():
        x_logits, tscams = model(x.unsqueeze(0).cuda(), True)

    x_probs = F.softmax(x_logits, dim=-1)
    pred_cls_id = x_probs.argmax()  # hardcode this to screw?

    cam_pred = tscams[0, pred_cls_id, :, :].detach().cpu().numpy()
    mask_pred = cv2.resize(cam_pred, im.size)
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask_pred = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)
    # mask_image = (mask_pred[..., np.newaxis] * im).astype("uint8")
    plt.axis('off')
    plt.imsave('../output/object_0_mask_pred.JPEG', mask_pred)
    plt.cla()
    plt.clf()
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

    ax1.set_title('Input Image')
    ax2.set_title('Token-Semantic Coupled Attention Map')
    ax3.set_title('Binary Map')

    _, mask_pred_binary_map = cv2.threshold(mask_pred,
                                            mask_pred.max() * 0.12, 1,
                                            cv2.THRESH_TOZERO)
    contours, _ = cv2.findContours((mask_pred_binary_map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    w, h = im.size
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
    heatmap = cv2.applyColorMap((mask_pred * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blend = np.array(im) * 0.5 + heatmap * 0.5
    x1, y1, x2, y2 = estimated_bbox
    im_box = cv2.rectangle(np.array(im), (x1, y1), (x2, y2), color1, 2)
    _ = ax1.imshow(im_box)  # Visualize Input Image with Estimated Box
    _ = ax2.imshow(mask_pred)  # Visualize TS-CAM which is localization map for estimating object box
    _ = ax3.imshow(mask_pred_binary_map)  # Visualize Binary Map
    plt.savefig('../output/object_0_box_pred.JPEG')

if __name__ == "__main__":
    main()