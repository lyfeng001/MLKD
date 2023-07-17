# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

import math
from matplotlib import pyplot as plt
from pysot.models.enhance_model import darklighter


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135', type=str,
        help='datasets')
parser.add_argument('--config', default='/home/user/V4R/LYF/pysot-master2/experiments/siamrpn_alex_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/home/user/V4R/LYF/pysot-master2/snapshot/checkpoint_e1.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
##tea
    img = cv2.imread("/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135/data_seq/pedestrian2/00575.jpg")
    print(img)
    img = torch.tensor(img).permute(2,0,1)
    img = img.reshape(1,3,1080,1920)
    enhancer = darklighter.DarkLighter().cuda()
    enhancer.load_state_dict(torch.load(
            '/home/user/V4R/LYF/pysot-master2/pysot/models/enhance_model/snapshots/Epoch193.pth'))
    img, _, _ = enhancer(img)
    img = img.reshape(3,1080,1920).permute(1,2,0)
    img = np.uint8(img.detach().cpu().numpy().copy())

    # max_n = img.max()
    # min_n = img.min()
    # img = 255*(img-min_n)/(max_n-min_n)
    # img = np.asarray(img.astype(np.uint8), order="C")
    # img = cv2.imread('/home/user/V4R/LYF/pysot-master2/tools/draw/result_pic/1.png')
##tea
    gt_bbox = [689.6137605925454,728.1901041846006,114.8810883385334,321.34633641704346]
    cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])),(int(gt_bbox[0])+int(gt_bbox[2]), int(gt_bbox[1])+int(gt_bbox[3])), (0, 0, 255), 8)
##spa
    gt_bbox1 = [856.7904866922415,727.2778143266784,42.97837266460963,139.81902801271352]
    cv2.rectangle(img, (int(gt_bbox1[0]), int(gt_bbox1[1])),(int(gt_bbox1[0])+int(gt_bbox1[2]), int(gt_bbox1[1])+int(gt_bbox1[3])), (255, 255, 0), 8)
##mcl
    gt_bbox2 = [856.7813424380435,728.4820529928671,43.78735310456317,140.85868773083058]
    cv2.rectangle(img, (int(gt_bbox2[0]), int(gt_bbox2[1])),(int(gt_bbox2[0])+int(gt_bbox2[2]), int(gt_bbox2[1])+int(gt_bbox2[3])), (255, 0, 255), 8)
##L2
    gt_bbox3 = [855.867382043132,727.9610641600038,43.97888681390802,138.78199319171298]
    cv2.rectangle(img, (int(gt_bbox3[0]), int(gt_bbox3[1])),(int(gt_bbox3[0])+int(gt_bbox3[2]), int(gt_bbox3[1])+int(gt_bbox3[3])), (0, 255, 255), 8)
    cv2.putText(img, "pedestrian2 #575", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     plt.imshow(img)
#     plt.savefig("/home/user/V4R/LYF/pysot-master2/tools/draw/result_pic/pedestrian2.png")
    cv2.imwrite("/home/user/V4R/LYF/pysot-master2/tools/draw/result_pic/pedestrian2.png",img)
    # cv2.imshow('1', img)
    # cv2.waitKey(1)


if __name__ == '__main__':
    main()
