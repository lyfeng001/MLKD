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
# from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pysot.tracker.utils import visualize_cam


import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

import math

from gradcam import GradCAM



parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135', type=str,
        help='datasets')
parser.add_argument('--config', default='/home/user/V4R/LYF/pysot-master2/experiments/siamrpn_alex_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/home/user/V4R/LYF/pysot-master2/snapshot_mcl/checkpoint_e2.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='basketballplayer2', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


    # grid_image = make_grid(zf2, normalize=True, scale_each=True)

    # 将网格图像张量转换为 numpy 数组，并交换通道维度顺序，以适应 Matplotlib 的要求
	# grid_image = grid_image.detach().cpu().numpy()

    # 使用 Matplotlib 将特征图显示为灰度图或热度图
    # plt.imshow(grid_image[:,:,0], cmap='gray')
    # 灰度图
    # for i in range(256):

        # plt.imshow(grid_image[i,:,:], cmap='jet')
    # plt.imshow(grid_image[0,:,:], cmap='hot')
    # plt.imshow(grid_image[0,:,:], cmap='coolwarm')

    # plt.colorbar()# 热度图
        # plt.show()
        # plt.savefig('/home/user/V4R/LYF/pysot-master2/toolkit/visualization/output/4z-'+str(i)+'.png', dpi=900, bbox_inches='tight')
	# cls, loc, cls_feature, loc_feature = modell1.rpn_head(zf, xf)

	# model_dict = dict(type='apn', arch=model1, layer_name='rpn_head.cls.conv_search', input_size=(512,512))
	# # print(model_dict.keys())
	# cam = GradCAM(model_dict)
	# data={}
	# data['search'] = search
	# data['template'] = template
	# maps, logit = cam(data, class_idx=1)
# Copyright (c) SenseTime. All Rights Reserved.




class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()


    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox,multi_stu=False):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        if multi_stu==False:
            self.model.template(z_crop)
        self.rpn_model_dict = dict(type='rpn', arch=self.model, layer_name='rpn_head.cls.conv_search.1', input_size=(287, 287))
        self.cam = GradCAM(self.rpn_model_dict, True)

    def track(self, img,idx,multi_stu=False,fea=None,new_score=None,new_bbox=None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # print('###############################\n\n\n\n\n\n\n')
        # print(self.size[0])
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)


        images = []
        res=[]

        mask,_=self.cam(x_crop,class_idx=1)

        mask_d, im_d=self.return_subwindows(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average,mask)
        # print(im_d)
        heatmap, result = visualize_cam(mask_d, im_d)
        images.append(torch.stack([result], 0))
        res.append(torch.stack([result],0))
        images = make_grid(torch.cat(images, 0), nrow=1)
        res = make_grid(torch.cat(res, 0), nrow=1)

        video_name = 'basketballplayer2_c0_4'

        output_dir = '/home/user/V4R/LYF/pysot-master2/tools/draw/SiamRPN++/{}/'.format(video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_name = str(idx)+'_cls.jpeg'

        output_path = os.path.join(output_dir, output_name)

        save_image(result, output_path)


        outputs = self.model.track(x_crop,multi_stu,fea)

        score = self._convert_score(outputs['cls'])
        if new_score is not None:
            score = new_score
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        if new_bbox is not None:
            pred_bbox = new_bbox

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track2(x_crop,multi_stu,fea)
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

    


TRACKS = {
          'SiamRPNTracker': SiamRPNTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)



def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = '/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135'

    # # create model
    # model = ModelBuilder(option=False,multi_stu=True)

    # # load model
    # model = load_pretrain(model, args.snapshot, option='trained').cuda().eval()

    modell1 = ModelBuilder(option=False)
    modell2 = ModelBuilder(option=False)
    modell3 = ModelBuilder(option=False)
    modell4 = ModelBuilder(option=True)
    model1 = load_pretrain(modell1, cfg.TRAIN.PRETRAINED_L2,option='trained').cuda().eval()
    model2 = load_pretrain(modell2, cfg.TRAIN.PRETRAINED_SPA,option='trained').cuda().eval()
    model3 = load_pretrain(modell3, cfg.TRAIN.PRETRAINED_MCL,option='trained').cuda().eval()
    model4 = load_pretrain(modell4, cfg.TRAIN.PRETRAINED_TEA,option='trained').cuda().eval()

    # model_dict = model1.state_dict()

    # for name, param in model1.named_parameters():
    #     print(name)
    #     print("requires_grad:", param.requires_grad)
    #     print("-----------------------------------")

    

    search = cv2.imread('/home/user/V4R/LYF/pysot-master2/tools/paper/origin/000000.31.x.jpg', 1).astype(np.uint8)
    template = cv2.imread('/home/user/V4R/LYF/pysot-master2/tools/paper/origin/000000.31.z.jpg', 1).astype(np.uint8)

    # img = img / 255.
    search = torch.from_numpy(search)
    search = torch.tensor(search, dtype=torch.float32)
    search = search.unsqueeze(0) #在第一维度上增加一个维度，作为batch size大小
    search = search.permute(0, 3, 1, 2).cuda()

    template = torch.from_numpy(template)
    template = torch.tensor(template, dtype=torch.float32)
    template = template.unsqueeze(0) #在第一维度上增加一个维度，作为batch size大小
    template = template.permute(0, 3, 1, 2).cuda()
    # build tracker
    # tracker = build_tracker(model)
    tracker1 = build_tracker(model1)
    tracker2 = build_tracker(model2)
    tracker3 = build_tracker(model3)
    tracker4 = build_tracker(model4)

    tracker = tracker4

    print(template.shape)
    zf = model1.backbone(template)
    xf = model1.backbone(search)
    print(zf.shape)
    print(xf.shape)
    zf2 = zf.reshape([256,6,6])
    xf2 = xf.reshape([256,54,54])
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
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)

                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img ,idx)
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
    # print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))
    print("done")

    dataset_name = 'UAVDark135'

    fps_path = os.path.join('/home/user/V4R/LYF/pysot-master2/fps', '{}.txt'.format(model_name))
    # with open(fps_path, 'w') as f:
    #     f.write('Time:{:5.1f},Speed:{:3.1f}'.format(TOC, IDX / TOC))
 


if __name__ == '__main__':
    main()
