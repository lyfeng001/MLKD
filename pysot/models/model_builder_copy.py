# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.integration import Integration
from pysot.models.head.rpn import DepthwiseXCorr

from sklearn import preprocessing as pp
import torch
import warnings

warnings.filterwarnings('ignore')

class ModelBuilder(nn.Module):
    def __init__(self,option,multi_stu=False):
        super(ModelBuilder, self).__init__()
        if multi_stu == False:
            if option:
                # build backbone
                self.backbone = get_backbone(cfg.KD.BACKBONE_TYPE,
                                            **cfg.BACKBONE.KWARGS)  # 将cfg文件中的backbone类型参数以及其他关键字参数输入，其中关键字参数为字典类型，通过**解包

                # build adjust layer
                if cfg.ADJUST.ADJUST:
                    self.neck = get_neck(cfg.KD.ADJUST.TYPE,
                                        **cfg.ADJUST.KWARGS)

                # build rpn head
                self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                            **cfg.RPN.KWARGS)

                # build mask head
                if cfg.MASK.MASK:
                    self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                                **cfg.MASK.KWARGS)

                    if cfg.REFINE.REFINE:
                        self.refine_head = get_refine_head(cfg.REFINE.TYPE)  # 下面的这些模块也与backbone同理

            else:

                # build backbone
                self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                            **cfg.BACKBONE.KWARGS)   #将cfg文件中的backbone类型参数以及其他关键字参数输入，其中关键字参数为字典类型，通过**解包

                # build adjust layer
                if cfg.ADJUST.ADJUST:
                    self.neck = get_neck(cfg.ADJUST.TYPE,
                                        **cfg.ADJUST.KWARGS)

                # build rpn head
                self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                            **cfg.RPN.KWARGS)

                # build mask head
                if cfg.MASK.MASK:
                    self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                                **cfg.MASK.KWARGS)

                    if cfg.REFINE.REFINE:
                        self.refine_head = get_refine_head(cfg.REFINE.TYPE)         #下面的这些模块也与backbone同理

        else:

            # # build backbone
            # self.backbone = get_backbone(cfg.BACKBONE.TYPE,
            #                              **cfg.BACKBONE.KWARGS)   #将cfg文件中的backbone类型参数以及其他关键字参数输入，其中关键字参数为字典类型，通过**解包

            # # build adjust layer
            # if cfg.ADJUST.ADJUST:
            #     self.neck = get_neck(cfg.ADJUST.TYPE,
            #                          **cfg.ADJUST.KWARGS)

            self.integration = Integration()

            # build rpn head
            self.rpn_head_cls = DepthwiseXCorr(256, 256, 10)
            self.rpn_head_loc = DepthwiseXCorr(256, 256, 20)
            
            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                               **cfg.MASK.KWARGS)

                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z, multi_stu=False):
        if multi_stu==False:
            zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        if multi_stu==False:
            self.zf = zf

    def track(self, x, multi_stu=False,fea=None):

        if multi_stu==False:
            xf = self.backbone(x)
            if cfg.MASK.MASK:
                self.xf = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            cls, loc, cls_feature, loc_feature = self.rpn_head(self.zf, xf)
            if cfg.MASK.MASK:
                mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        else:
            fea_done_loc = self.integration(fea123_loc)
            fea_done_cls = self.integration(fea123_cls)
            loc = self.rpn_head_loc.get_headout(fea_done_loc)
            cls = self.rpn_head_cls.get_headout(fea_done_cls)
            cls_feature = fea_done_cls
            loc_feature = fea_done_loc
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }


    def track2(self, x, multi_stu=False,fea=None):

        
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc, cls_feature, loc_feature = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        
        return {
                'cls': cls,
                'loc': loc,
                'feature_cls': cls_feature,
                'feature_loc': loc_feature,
                'mask': mask if cfg.MASK.MASK else None
               }


    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data, teacher_model=None,multi_stu=False,get_feature=False,test_mode=False,heatmap_mode=False):
        """ only used in training
        """
        if heatmap_mode == False:

            if multi_stu:
                fea123_cls = data['fea123_cls']
                fea123_loc = data['fea123_loc']

            if test_mode:
                pass
            elif heatmap_mode:
                template = data['template'].cuda()
                search = data['search'].cuda()
            else:
                template = data['template'].cuda()
                search = data['search'].cuda()
                label_cls = data['label_cls'].cuda()
                label_loc = data['label_loc'].cuda()
                label_loc_weight = data['label_loc_weight'].cuda()


            # get feature
            if multi_stu == False:
                zf = self.backbone(template)
                xf = self.backbone(search)
                if cfg.MASK.MASK:
                    zf = zf[-1]
                    self.xf_refine = xf[:-1]
                    xf = xf[-1]
                if cfg.ADJUST.ADJUST:
                    zf = self.neck(zf)
                    xf = self.neck(xf)
                cls, loc, cls_feature, loc_feature = self.rpn_head(zf, xf)

                mul_feature_cls = cls_feature
                mul_feature_loc = loc_feature
        else:
            xf = self.backbone(data)
            cls, loc, cls_feature, loc_feature = self.rpn_head(self.zf, xf)
            cls = self.log_softmax(cls)

        if get_feature :
            return mul_feature_loc, mul_feature_cls
        elif heatmap_mode:
            return cls
        
        else:


            tea_zf = teacher_model.backbone(template)
            tea_xf = teacher_model.backbone(search)
            tea_cls, tea_loc, tea_cls_feature, tea_loc_feature = teacher_model.rpn_head(tea_zf, tea_xf)

            if multi_stu:
                fea_done_cls = self.integration(fea123_cls)
                fea_done_loc = self.integration(fea123_loc)
                loc = self.rpn_head_loc.get_headout(fea_done_loc)
                cls = self.rpn_head_cls.get_headout(fea_done_cls)
                cls_feature = fea_done_cls
                loc_feature = fea_done_loc

            ### kd loss
            soft_loss = nn.KLDivLoss(reduction='batchmean')
            Tem = cfg.KD.TEM
            
            b, a2, h, w = cls.size()
            clss = cls
            tea_clss = tea_cls

            locc = loc
            tea_locc = tea_loc
            clss =  clss.view(b, 2, a2//2, h, w).permute(0, 2, 3, 4, 1).reshape(-1,2).contiguous()
            tea_clss = tea_clss.view(b, 2, a2//2, h, w).permute(0, 2, 3, 4, 1).reshape(-1,2).contiguous()


            locc = locc.permute(0,2,3,1).reshape(-1,4*cfg.ANCHOR.ANCHOR_NUM).contiguous()
            tea_locc = tea_locc.permute(0,2,3,1).reshape(-1,4*cfg.ANCHOR.ANCHOR_NUM).contiguous()
            
            
            kd_loss = (soft_loss(F.log_softmax(clss/Tem,dim=0),F.softmax(tea_clss/Tem,dim=0))+ \
                        soft_loss(F.log_softmax(locc/Tem,dim=0),F.softmax(tea_locc/Tem,dim=0)))
            # print('cls=\n')

            # print('###################################')
            # print(cls_feature.size())
            # print('#####################################')
            # print('11111111111111111111111111111111111')
            # print(loc_feature.size())
            # print('11111111111111111111111111111111111')
            # ### corr_loss

            cls_feature1 = cls_feature
            cls_feature2 = cls_feature
            cls_feature3 = cls_feature
            tea_cls_feature1 = tea_cls_feature
            tea_cls_feature2 = tea_cls_feature
            tea_cls_feature3 = tea_cls_feature

    ##################################
    #########  L2 Loss  ##############
    ##################################

            L2_loss = nn.MSELoss()
            cls_feature1 = cls_feature1.reshape(-1)
            tea_cls_feature1 = tea_cls_feature1.reshape(-1)
            cls_feature1 = F.softmax(cls_feature1)
            tea_cls_feature1 = F.softmax(tea_cls_feature1)
            corr_loss1 = L2_loss(cls_feature1,tea_cls_feature1)*0.01
                
    ##################################
    #########  Spa Loss  #############
    ##################################

            cls_feature2 = cls_feature2.reshape(1,-1,16*cls_feature2.size(3),16*cls_feature2.size(3))
            tea_cls_feature2 = tea_cls_feature2.reshape(1,-1,16*tea_cls_feature2.size(3),16*tea_cls_feature2.size(3))

            kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
            kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
            kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
            kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
            weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
            weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
            weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
            weight_down = nn.Parameter(data=kernel_down, requires_grad=False)


            b,c,h,w = tea_cls_feature2.shape

            tea_mean = torch.mean(tea_cls_feature2,1,keepdim=True)
            stu_mean = torch.mean(cls_feature2,1,keepdim=True)

            D_tea_left = F.conv2d(tea_mean , weight_left, padding=1)
            D_tea_right = F.conv2d(tea_mean , weight_right, padding=1)
            D_tea_up = F.conv2d(tea_mean , weight_up, padding=1)
            D_tea_down = F.conv2d(tea_mean , weight_down, padding=1)

            D_stu_letf = F.conv2d(stu_mean , weight_left, padding=1)
            D_stu_right = F.conv2d(stu_mean , weight_right, padding=1)
            D_stu_up = F.conv2d(stu_mean , weight_up, padding=1)
            D_stu_down = F.conv2d(stu_mean , weight_down, padding=1)

            D_left = torch.pow(D_tea_left - D_stu_letf,2)
            D_right = torch.pow(D_tea_right - D_stu_right,2)
            D_up = torch.pow(D_tea_up - D_stu_up,2)
            D_down = torch.pow(D_tea_down - D_stu_down,2)
            
            corr_loss2 = torch.mean((D_left + D_right + D_up +D_down))

    ########################################
    #########  Multi-stu Loss  #############
    ########################################

            L2_loss = nn.MSELoss()
            cls_feature3 = cls_feature3.reshape(-1)
            tea_cls_feature3 = tea_cls_feature3.reshape(-1)
            cls_feature3 = F.softmax(cls_feature3)
            tea_cls_feature3 = F.softmax(tea_cls_feature3)

            tea_mean = torch.mean(tea_cls_feature3)
            stu_mean = torch.mean(cls_feature3)

            zero = torch.zeros_like(tea_cls_feature3)
            one = torch.ones_like(tea_cls_feature3)

            tea_feature_mul = torch.where(tea_cls_feature3>tea_mean, one, zero)
            stu_feature_mul = torch.where(cls_feature3>stu_mean, one, zero)

            corr_loss3 = L2_loss(stu_feature_mul, tea_feature_mul)*5




            # get loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
            tea_cls_loss = select_cross_entropy_loss(tea_cls, label_cls)
            tea_loc_loss = weight_l1_loss(tea_loc, label_loc, label_loc_weight)
            outputs = {}

            if cls_loss < tea_cls_loss and loc_loss < tea_loc_loss:
                outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss 

            if cls_loss >= tea_cls_loss and loc_loss < tea_loc_loss:
                outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss + soft_loss(F.log_softmax(clss/Tem,dim=0),F.softmax(tea_clss/Tem,dim=0))

            if cls_loss < tea_cls_loss and loc_loss >= tea_loc_loss:
                outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss + soft_loss(F.log_softmax(locc/Tem,dim=0),F.softmax(tea_locc/Tem,dim=0))

            if cls_loss >= tea_cls_loss and loc_loss >= tea_loc_loss:
                outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.KD_WEIGHT * kd_loss + cfg.TRAIN.CORR_WEIGHT * (corr_loss1+corr_loss2+corr_loss3)



            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
            outputs['kd_loss'] = kd_loss
            if multi_stu==False:
                outputs['corr_loss'] = corr_loss1+corr_loss2+corr_loss3




            if cfg.MASK.MASK:
                # TODO
                mask, self.mask_corr_feature = self.mask_head(zf, xf)
                mask_loss = None
                outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
                outputs['mask_loss'] = mask_loss
			
        
            return outputs
