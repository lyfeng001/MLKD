# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

#进来先执行这里，获得配置文件等
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='../experiments/siamrpn_alex_dwxcorr_16gpu/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()
global rank, world_size, inited

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)       #若为分布式训练 则使用sampler进行数据的分配


    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)        #载入数据
    return train_loader


def build_opt_lr(model, current_epoch=0,multi_stu=False):
    # for param in model.backbone.parameters():
    #     param.requires_grad = False             #首先冻结梯度，防止参数变化
    # for m in model.backbone.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()                            #首先需要将BN层转化为测试模式，这是因为train时BN需要当前batch的均值与方差，同时也需要
                                                #之前batch的，而测试时只需要历史的，以防止batchsize发生变化
    # if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
    #     for layer in cfg.BACKBONE.TRAIN_LAYERS:
    #         for param in getattr(model.backbone, layer).parameters():
    #             param.requires_grad = True
    #         for m in getattr(model.backbone, layer).modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.train()
    if multi_stu:
        trainable_params = []
        # trainable_params += [{'params': filter(lambda x: x.requires_grad,
        #                                        model.backbone.parameters()),
        #                       'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

        if cfg.ADJUST.ADJUST:
            trainable_params += [{'params': model.neck.parameters(),
                                'lr': cfg.TRAIN.BASE_LR}]

        # for param in model.backbone.parameters():
        #     param.requires_grad = False

        # for param in model.rpn_head_loc.parameters():
        #     param.requires_grad = False
        # for param in model.rpn_head_cls.parameters():
        #     param.requires_grad = False
        for param in model.integration.parameters():
            param.requires_grad = True
        for param in model.rpn_head_cls.parameters():
            param.requires_grad = False
        for param in model.rpn_head_loc.parameters():
            param.requires_grad = False

        # trainable_params += [{'params': filter(lambda x: x.requires_grad,
        #                                    model.rpn_head_loc.parameters()),
        #                                 'lr': cfg.TRAIN.BASE_LR}]
        # trainable_params += [{'params': filter(lambda x: x.requires_grad,
        #                                    model.rpn_head_cls.parameters()),
        #                                 'lr': cfg.TRAIN.BASE_LR}]
        # trainable_params += [{'params': filter(lambda x: x.requires_grad,
        #                                        model.rpn_head.parameters()),
        #                       'lr': cfg.TRAIN.BASE_LR}]

        
        if cfg.MASK.MASK:
            trainable_params += [{'params': model.mask_head.parameters(),
                                'lr': cfg.TRAIN.BASE_LR}]

        if cfg.REFINE.REFINE:
            trainable_params += [{'params': model.refine_head.parameters(),
                                'lr': cfg.TRAIN.LR.BASE_LR}]
        
        trainable_params += [{'params': model.integration.parameters(),
                            'lr': cfg.TRAIN.BASE_LR}]

        optimizer = torch.optim.SGD(trainable_params,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
        lr_scheduler.step(cfg.TRAIN.START_EPOCH)
        return optimizer, lr_scheduler

    else:
        for param in model.backbone.parameters():
            param.requires_grad = False             #首先冻结梯度，防止参数变化
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()                            #首先需要将BN层转化为测试模式，这是因为train时BN需要当前batch的均值与方差，同时也需要
                                                    #之前batch的，而测试时只需要历史的，以防止batchsize发生变化
        if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
            for layer in cfg.BACKBONE.TRAIN_LAYERS:
                for param in getattr(model.backbone, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.backbone, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()

        trainable_params = []
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                            model.backbone.parameters()),
                            'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

        if cfg.ADJUST.ADJUST:
            trainable_params += [{'params': model.neck.parameters(),
                                'lr': cfg.TRAIN.BASE_LR}]


        for param in model.rpn_head.cls.head.parameters():
            param.requires_grad = False
        for param in model.rpn_head.loc.head.parameters():
            param.requires_grad = False

        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                            model.rpn_head.parameters()),
                            'lr': cfg.TRAIN.BASE_LR}]

        if cfg.MASK.MASK:
            trainable_params += [{'params': model.mask_head.parameters(),
                                'lr': cfg.TRAIN.BASE_LR}]

        if cfg.REFINE.REFINE:
            trainable_params += [{'params': model.refine_head.parameters(),
                                'lr': cfg.TRAIN.LR.BASE_LR}]

        optimizer = torch.optim.SGD(trainable_params,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
        lr_scheduler.step(cfg.TRAIN.START_EPOCH)
        return optimizer, lr_scheduler




def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        # tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
        #                      _norm, tb_index)
        # tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
        #                      w_norm, tb_index)
        # tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
        #                      w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    # tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    # tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    # tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer,teacher,teacher_model=None,multi_stu=False):
    cur_lr = lr_scheduler.get_cur_lr()      #获得当前的学习率值
    rank = 0#get_rank()                       #获得机器的编号

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)  #判断数字是否有效

    world_size = get_world_size()           #共有几台机器
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch                     #第几轮开始训练

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR): #and \
            # get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()                                             #结束时间
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch          #修正epoch值用于输出显示

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:       #当轮数到一定值之后开始训练backbone
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch,multi_stu) #获得新的优化器以及lr数值
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    pass
                    # tb_writer.add_scalar('lr/group{}'.format(idx+1),
                    #                      pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            pass
            # tb_writer.add_scalar('time/data', data_time, tb_idx)
        
        if multi_stu:
            modell1 = ModelBuilder(option=False)
            modell2 = ModelBuilder(option=False)
            modell3 = ModelBuilder(option=False)
            model1 = load_pretrain(modell1, cfg.TRAIN.PRETRAINED_L2,option='trained').cuda().eval()
            model2 = load_pretrain(modell2, cfg.TRAIN.PRETRAINED_SPA,option='trained').cuda().eval()
            model3 = load_pretrain(modell3, cfg.TRAIN.PRETRAINED_MCL,option='trained').cuda().eval()
            fea1_loc,fea1_cls = model1(data,teacher_model,get_feature=True)
            fea2_loc,fea2_cls = model2(data,teacher_model,get_feature=True)
            fea3_loc,fea3_cls = model3(data,teacher_model,get_feature=True)
            data['fea123_loc'] = torch.cat([fea1_loc,fea2_loc,fea3_loc],dim=1)
            data['fea123_cls'] = torch.cat([fea1_cls,fea2_cls,fea3_cls],dim=1)

            outputs = model(data,teacher_model,multi_stu=True)

        else:
            if teacher:
                outputs = model(data)           #调用model中的forward函数，获得输出
            else:
                outputs = model(data,teacher_model)

        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            reduce_gradients(model)         #用于分布式训练梯度相加

            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                pass
                # log_grads(model.module, tb_writer, tb_idx)      #将tensorboard、数据编号传入

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)        #梯度剪裁
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)         #获得每一个batch的用时以及整体用时
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                
                pass
                # tb_writer.add_scalar(k, v, tb_idx)
                

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()

    return model



def select_train(rank, world_size, option,multi_stu=False):

    # create model
    if option:
        model = ModelBuilder(option).cuda().train()  #进入模型建立这一函数进行模型的建立，并且设置模式为训练模式
    else:
        model = ModelBuilder(option,multi_stu=True).cuda().train()  #进入模型建立这一函数进行模型的建立，并且设置模式为训练模式

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)  #需要的backbone的参数文件的相对路径
        load_pretrain(model.backbone, backbone_path)    #通过这一函数加载模型参数

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = None
        # tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        # print('##################################################\n\n\n\n')
        # print(tb_writer)
    else:
        tb_writer = None
        # print('********************************************************111111111')
    # build dataset loader
    train_loader = build_data_loader()      #建立dataloder加载数据

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH,multi_stu)       #将模型与开始训练的轮数传入，相当于优化器和lr的初始化操作

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        if option == True:
            model = load_pretrain(model, cfg.TRAIN.PRETRAINED_TEA,option='trained').cuda()
        else:
            pretrain_dict = torch.load(cfg.TRAIN.PRETRAINED_SPA)['state_dict']
            model_dict = model.state_dict()
            new_pre_dict = {k:v for k,v in pretrain_dict.items() if 'rpn_head' in k}
            new_model_dict = {k:v for k,v in model_dict.items() if ('rpn_head_cls' in k) or ('rpn_head_loc' in k)}
            
            for i,key in enumerate(new_pre_dict):
                for k,key2 in enumerate(new_model_dict):
                    if k==i:
                        new_model_dict[key2] = new_pre_dict[key]

            model.load_state_dict(new_model_dict,strict=False)
            # model = load_pretrain(model, cfg.TRAIN.PRETRAINED_SPA, option='trained').cuda()

        # model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED,
        #     map_location=lambda storage, loc: storage.cpu()))
    return model, tb_writer, train_loader, optimizer, lr_scheduler





def main():
    rank, world_size = dist_init()  #用于分布式训练的

    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)       #通过访问arg中定义的cfg文件地址，获得配置文件信息
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)          #创建日志文件
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))


    option = cfg.KD.TEACHER_USED

    teacher_model, _, __, ___, ____ = select_train(rank, world_size, option)

    # tea_dist_model = DistModule(teacher_model)                          #加载预训练模型
    #
    # logger.info(lr_scheduler)
    # logger.info("teacher model prepare done")
    #
    # # start training
    # teacher_model = train(train_loader, tea_dist_model, optimizer, lr_scheduler, tb_writer,teacher==True)

    # 将loader、分布模型、优化器、学习率调节器、tensorboard均属入train函数中

    model, tb_writer, train_loader, optimizer, lr_scheduler = select_train(rank, world_size,option=False,multi_stu=True)

    dist_model = DistModule(model)  # 加载预训练模型

    logger.info(lr_scheduler)
    logger.info("model prepare done")
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer, teacher=False,teacher_model=teacher_model,multi_stu=True)




if __name__ == '__main__':
    seed_torch(args.seed)   #通过arg将种子传入seed设置函数中

    # a = os.environ



    # rank = 0
    # world_size = 1
    # inited = True


    main()  #进主函数
