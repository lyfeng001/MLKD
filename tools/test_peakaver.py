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

 
import matplotlib.pyplot as plt
import scipy.signal as sg

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
parser.add_argument('--snapshot', default='/home/user/V4R/LYF/pysot-master2/snapshot/checkpoint_e3.pth', type=str,
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

                #读取图片
                # source1 = cv2.cvtColor(score1,cv2.COLOR_BGR2RGB)
                # source2 = cv2.cvtColor(score2,cv2.COLOR_BGR2RGB)
                # source3 = cv2.cvtColor(score3,cv2.COLOR_BGR2RGB)

                #均值滤波
                # result1 = cv2.blur(source1, (5,5))
                # result2 = cv2.blur(source2, (5,5))
                # result3 = cv2.blur(source3, (5,5))
                
                # print(score1)

                max_index1 = torch.tensor(sg.argrelmax(np.array(score1)))
                max_index2 = torch.tensor(sg.argrelmax(np.array(score2)))
                max_index3 = torch.tensor(sg.argrelmax(np.array(score3)))



                value1 = torch.tensor(score1[max_index1]).cuda().reshape(-1)
                value2 = torch.tensor(score2[max_index2]).cuda().reshape(-1)
                value3 = torch.tensor(score3[max_index3]).cuda().reshape(-1)

                value1.sort()
                value2.sort()
                value3.sort()


                weight1 = (value1[-1]/value1[-2]).cpu().detach().numpy()
                weight2 = (value2[-1]/value2[-2]).cpu().detach().numpy()
                weight3 = (value3[-1]/value3[-2]).cpu().detach().numpy()

                # new_bbox = (pred_bbox1+pred_bbox2+pred_bbox3)/3
                # new_score = (score1+score2+score3)/3

                new_bbox = (pred_bbox1*weight1+pred_bbox2*weight2+pred_bbox3*weight3)/(weight1+weight2+weight3)
                new_score = (score1*weight1+score2*weight2+score3*weight3)/(weight1+weight2+weight3)

                '''if idx == 5:
                    # print(new_bbox.shape)
                    new = new_score.reshape(49,45)
                    score1 = score1.reshape(49,45)
                    score2 = score2.reshape(49,45)
                    score3 = score3.reshape(49,45)
                
                    # x = [i for i in range(2205)]
                    # plt.plot(x, score1, alpha=0.5, linewidth=1)
                    # plt.xticks([])  #去掉横坐标值
                    # plt.yticks([])
                    # plt.axis('off')
                    # plt.show
                 
                    print(outputs1['cls'].shape)
                    new_cls = outputs3['cls'].reshape(10,21,21)
                    new_cls1 = outputs1['cls'].reshape(10,21,21)
                    new_cls2 = outputs2['cls'].reshape(10,21,21)
                    new_cls = new_cls.detach().cpu().numpy()
                    new_cls1 = new_cls1.detach().cpu().numpy()
                    new_cls2 = new_cls2.detach().cpu().numpy()
                    new = new_cls*weight1+new_cls1*weight2+new_cls2*weight3
                    plt.matshow(new[0,:,:],cmap='jet')
                    plt.xticks([])  
                    plt.yticks([])
                    plt.axis('off')
                    np.save("/home/user/V4R/LYF/pysot-master2/tools/draw/score/1.npy",new_cls)
                    np.save("/home/user/V4R/LYF/pysot-master2/tools/draw/score/2.npy",new_cls1)
                    np.save("/home/user/V4R/LYF/pysot-master2/tools/draw/score/3.npy",new_cls2)
                    np.save("/home/user/V4R/LYF/pysot-master2/tools/draw/score/4.npy",new)
                   

                    new3_cls = new_cls2[0,:,:]  #.reshape(1,21,21).permute(1,2,0)
                    fig = plt.figure()
                    ax = plt.axes(projection="3d")
                    x = y = np.arange(start=0, stop=20, step=1)
                    X, Y = np.meshgrid(x, y)
                    Z = new3_cls[X,Y]
                    ax.plot_surface(X,Y,Z,cmap='jet')
                    plt.axis('off')
                    plt.show()
                    plt.savefig('/home/user/V4R/LYF/pysot-master2/tools/draw/score/3-3.png')
                     '''





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
