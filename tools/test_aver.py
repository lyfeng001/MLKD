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


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135', type=str,
        help='datasets')
parser.add_argument('--config', default='/home/user/V4R/LYF/pysot-master2/experiments/siamrpn_alex_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/home/user/V4R/LYF/pysot-master2/snapshot/checkpoint_e2.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = '/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135'

    # create model
    # model = ModelBuilder(option=False)

    modell1 = ModelBuilder(option=False)
    modell2 = ModelBuilder(option=False)
    modell3 = ModelBuilder(option=False)
    model1 = load_pretrain(modell1, cfg.TRAIN.PRETRAINED_L2,option='trained').cuda().eval()
    model2 = load_pretrain(modell2, cfg.TRAIN.PRETRAINED_SPA,option='trained').cuda().eval()
    model3 = load_pretrain(modell3, cfg.TRAIN.PRETRAINED_MCL,option='trained').cuda().eval()

    # load model
    # model = load_pretrain(model, args.snapshot, option='trained').cuda().eval()

    # build tracker
    # tracker = build_tracker(model)
    tracker1 = build_tracker(model1)
    tracker2 = build_tracker(model2)
    tracker3 = build_tracker(model3)

    # # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    IDX = 0
    TOC = 0
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker1.init(img, gt_bbox_)
                tracker2.init(img, gt_bbox_)
                tracker3.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)

                pred_bboxes.append(pred_bbox)
            else:
                outputs1 = tracker1.track2(img)
                outputs2 = tracker2.track2(img)
                outputs3 = tracker3.track2(img)
                score1 = tracker1._convert_score(outputs1['cls'])
                score2 = tracker2._convert_score(outputs2['cls'])
                score3 = tracker3._convert_score(outputs3['cls'])
                pred_bbox1 = tracker1._convert_bbox(outputs1['loc'], tracker1.anchors)
                pred_bbox2 = tracker2._convert_bbox(outputs2['loc'], tracker2.anchors)
                pred_bbox3 = tracker3._convert_bbox(outputs3['loc'], tracker3.anchors)

                
                new_bbox = (pred_bbox1+pred_bbox2+pred_bbox3)/3
                new_score = (score1+score2+score3)/3
                _ = tracker2.track(img, new_score=new_score, new_bbox=new_bbox)
                _ = tracker3.track(img, new_score=new_score, new_bbox=new_bbox)
                outputs = tracker1.track(img,new_score=new_score,new_bbox=new_bbox)
                
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if math.isnan(gt_bbox[1]):
                    gt_bbox = [0,0,0,0]
                else:
                    gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))
        IDX += idx
        TOC += toc
    print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))
    fps_path = os.path.join('results', dataset_name, '{}.txt'.format(model_name))
    with open(fps_path, 'w') as f:
        f.write('Time:{:5.1f},Speed:{:3.1f}'.format(TOC, IDX / TOC))


if __name__ == '__main__':
    main()
