# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
from pysot.core.config import cfg

class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def return_subwindows(self, im, pos, model_sz, original_sz, avg_chans, mask):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        # if any([top_pad, bottom_pad, left_pad, right_pad]):
        #     size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        #     te_im = np.zeros(size, np.uint8)
        #     te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        #     if top_pad:
        #         te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        #     if bottom_pad:
        #         te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        #     if left_pad:
        #         te_im[:, 0:left_pad, :] = avg_chans
        #     if right_pad:
        #         te_im[:, c + left_pad:, :] = avg_chans
        #     im_patch = te_im[int(context_ymin):int(context_ymax + 1),
        #                      int(context_xmin):int(context_xmax + 1), :]
        # else:
        #     im_patch = im[int(context_ymin):int(context_ymax + 1),
        #                   int(context_xmin):int(context_xmax + 1), :]
        mask_patch=mask[0,0,:,:].cpu().numpy()
        if not np.array_equal(model_sz, original_sz):
            mask_patch = cv2.resize(mask_patch, (int(original_sz), int(original_sz)))
        
        returned_mask=np.zeros((r,c),np.float32)
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad)
            te_im = np.zeros(size, np.float32)
            te_im[int(context_ymin):int(context_ymin)+int(original_sz),int(context_xmin):int(context_xmin)+int(original_sz)]=mask_patch
            returned_mask = te_im[top_pad:top_pad + r, left_pad:left_pad + c]
        else:
            returned_mask[int(context_ymin):int(context_ymin)+int(original_sz),int(context_xmin):int(context_xmin)+int(original_sz)]=mask_patch
        returned_im=im.transpose(2, 0, 1)
        returned_im=returned_im[np.newaxis,(2,1,0),:,:]
        returned_mask = returned_mask[np.newaxis, np.newaxis, :, :]
        # im_patch = im_patch.astype(np.float32)
        returned_mask = torch.from_numpy(returned_mask)
        returned_im = torch.from_numpy(returned_im)
        if cfg.CUDA:
            returned_mask = returned_mask.cuda()
            returned_im = returned_im.cuda()
        return returned_mask,returned_im
