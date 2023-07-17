import cv2
import os
import numpy as np

import torch
from matplotlib import pyplot as plt

from pysot.models.enhance_model import darklighter


def genafilepath(dataset, seq):
    if dataset in ['UAV123_10fps', 'UAV123_20L', 'UAVTrack112', 'DarkTrack2021', 'NAT2021', 'UAVDark135', 'UAVDT']:
        filepath = '/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135/data_seq/{}'.format(seq) #D:/Dataset/{}/data_seq/{}'.format(dataset, seq)
    elif dataset in ['VisDrone2018-SOT-test']:
        filepath = 'D:/Dataset/{}/sequences/{}'.format(dataset, seq)
    elif dataset in ['DTB70']:
        filepath = 'D:/Dataset/{}/{}/img'.format(dataset, seq)
    elif dataset in ['Onboard']:
        filepath = 'D:/Code/CDT_results/{}/data_seq/{}'.format(dataset, seq)
    return filepath

def genagtpath(dataset, seq):
    if dataset in ['UAV123_10fps', 'UAV123_20L', 'UAVTrack112', 'DarkTrack2021', 'NAT2021', 'UAVDark135']:
        gtpath = "/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135/anno/{}.txt".format(seq) #"D:/Dataset/{}/anno/{}.txt".format(dataset, seq)
    elif dataset in ['UAVDT']:
        gtpath = "D:/Dataset/{}/anno/{}_gt.txt".format(dataset, seq)
    elif dataset in ['VisDrone2018-SOT-test']:
        gtpath = 'D:/Dataset/{}/annotations/{}.txt'.format(dataset, seq)
    elif dataset in ['DTB70']:
        gtpath = 'D:/Dataset/{}/{}/groundtruth_rect.txt'.format(dataset, seq)
    elif dataset in ['Onboard']:
        gtpath = 'D:/Code/CDT_results/{}/anno/{}.txt'.format(dataset, seq)
    return gtpath

def genaimgsuffix(dataset, i):   
    if dataset in ['UAV123_10fps', 'UAV123_20L', 'DarkTrack2021', 'NAT2021']:
        imgsuffix = '/{}.jpg'.format(str('%06d' % (int(i)+1)))
    elif dataset in ['UAVTrack112', 'UAVDark135']:
        imgsuffix = '/{}.jpg'.format(str('%05d' % (int(i)+1)))
    elif dataset in ['UAVDT']:
        imgsuffix = '/img{}.jpg'.format(str('%06d' % (int(i)+1)))
    elif dataset in ['VisDrone2018-SOT-test']:
        imgsuffix = '/img{}.jpg'.format(str('%07d' % (int(i)+1)))
    elif dataset in ['DTB70']:
        imgsuffix = '/{}.jpg'.format(str('%05d' % (int(i)+1)))
    elif dataset in ['Onboard']:
        imgsuffix = '/{}.jpg'.format(str('%06d' % (int(i)+1)))
    return imgsuffix

def imgcrop(img, gt_bbox, i, w=720, h=480):
    yi = img.shape[0]
    xi = img.shape[1]
    cx, cy = int(float(gt_bbox[i][0])+float(gt_bbox[i][2])/2), int(float(gt_bbox[i][1])+float(gt_bbox[i][3])/2)
    y0, y1, x0, x1 = int(cy-h/2), int(cy+h/2), int(cx-w/2), int(cx+w/2)
    if y0<0:
        y0 = 0
        y1 = h
    if x0<0:
        x0 = 0
        x1 = w
    if y1>yi:
        y1 = yi
        y0 = yi - h
    if x1>xi:
        x1 = xi
        x0 = xi - w
    img = img[y0:y1, x0:x1]
    return img

def drawline(img,pt1,pt2,color,thickness=1,style='dashed',gap=10): 
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
    pts= [] 
    for i in np.arange(0,dist,gap): 
        r=i/dist 
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
        p = (x,y) 
        pts.append(p) 
 
    if style=='dotted': 
        for p in pts: 
            cv2.circle(img,p,thickness,color,-1) 
    else: 
        s=pts[0] 
        e=pts[0] 
        i=0 
        for p in pts: 
            s=e 
            e=p 
            if i%2==1: 
                cv2.line(img,s,e,color,thickness) 
            i+=1 


def drawrectangle(img, x0y0, x1y1, color=(255, 0, 0), thickness=1):
    drawline(img, (x0y0[0], x0y0[1]), (x1y1[0], x0y0[1]), color, thickness)
    drawline(img, (x0y0[0], x0y0[1]), (x0y0[0], x1y1[1]), color, thickness)
    drawline(img, (x1y1[0], x1y1[1]), (x1y1[0], x0y0[1]), color, thickness)
    drawline(img, (x1y1[0], x1y1[1]), (x0y0[0], x1y1[1]), color, thickness)


def main(dataset, seq, trackers, colors, crop):

    filepath = genafilepath(dataset, seq)
    gtpath = genagtpath(dataset, seq)
    if crop:
        resultpath = '/home/user/V4R/LYF/pysot-master2/tools/draw/{}/{}_crop'.format(dataset, seq)
    else:
        resultpath = '/home/user/V4R/LYF/pysot-master2/tools/draw/{}/{}'.format(dataset, seq)
    files = os.listdir(filepath)
    num = len(files)

    BBox = {}
    for tracker in trackers:
        BBox[tracker] = []
        # with open("D:/Code/CDT_results/{}/results/{}/{}.txt".format(dataset, tracker, seq), 'r') as box:
        with open("/home/user/V4R/LYF/pysot-master2/test_result/{}/{}.txt".format(tracker,seq),'r') as box:
            for i in range(num):
                BBox[tracker].append(box.readline()[:-1].split(','))

    gt_bbox = []
    with open(gtpath.format(dataset, seq), 'r') as gt:
        for i in range(num):
            l = gt.readline()[:-1].split(',')
            for i in range(4):
                if l[i]=='NaN' or l[i]=='nan':
                    l[i] = gt_bbox[-1][i]
            gt_bbox.append(l)

    if not os.path.isdir(resultpath):
                    os.makedirs(resultpath)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('./draw/R/{}.mp4'.format(seq),fourcc,30,(1920,1080))

    for i in range(num):
        img = cv2.imread(filepath + genaimgsuffix(dataset, i))
        
    ###  enhance  ###
        img = torch.tensor(img).permute(2,0,1)
        img = img.reshape(1,3,1080,1920)
        enhancer = darklighter.DarkLighter().cuda()
        enhancer.load_state_dict(torch.load(
                '/home/user/V4R/LYF/pysot-master2/pysot/models/enhance_model/snapshots/Epoch193.pth'))
        img, _, _ = enhancer(img)
        img = img.reshape(3,1080,1920).permute(1,2,0)
        img = np.uint8(img.detach().cpu().numpy().copy())


        ## cv2.rectangle(img, (int(float(gt_bbox[i][0])-5), int(float(gt_bbox[i][1]))),
        ##             (int(float(gt_bbox[i][0]))+int(float(gt_bbox[i][2])), int(float(gt_bbox[i][1]))+int(float(gt_bbox[i][3]))), (0, 255, 0), 4)

        c = 0
        for tracker in trackers:
            cv2.rectangle(img, (int(float(BBox[tracker][i][0])), int(float(BBox[tracker][i][1]))),
                        (int(float(BBox[tracker][i][0]))+int(float(BBox[tracker][i][2])), int(float(BBox[tracker][i][1]))+int(float(BBox[tracker][i][3]))), colors[c], 4)

            # if (c % 2) == 0:
            #     cv2.rectangle(img, (int(float(BBox[tracker][i][0])), int(float(BBox[tracker][i][1]))),
            #             (int(float(BBox[tracker][i][0]))+int(float(BBox[tracker][i][2])), int(float(BBox[tracker][i][1]))+int(float(BBox[tracker][i][3]))), colors[c], 4)
            # else:
            #     drawrectangle(img, (int(float(BBox[tracker][i][0])), int(float(BBox[tracker][i][1]))),
            #             (int(float(BBox[tracker][i][0]))+int(float(BBox[tracker][i][2])), int(float(BBox[tracker][i][1]))+int(float(BBox[tracker][i][3]))), colors[c], 3)
                
            c += 1

        if crop:
            img = imgcrop(img, gt_bbox, i)

        # cv2.imshow("video", img)

        # out.write(img)

        cv2.imwrite(resultpath + '/{}.jpg'.format(str('%06d' % (int(i)+1))), img)
        # cv2.waitKey(1)

    # out.release()
    
    cv2.destroyAllWindows()
    print('{}:{}'.format(dataset, seq))


if __name__ == '__main__':
    
    trackers = ['kdSiamRPN_tea','kdSiamRPN_stuspa_64','kdSiamRPN_stumcl_64','kdSiamRPN_stunew_64']
    colors = [[0,0,220],[220,220,0], [220,0,220], [0,220,220], [0,220,220],[220,0,0], [220,0,0], [0,0,220]]## red blue pink yellow
    crop = 1
    

    dataset = "UAVDark135"
    seq = "pedestrian7_1"
    main(dataset, seq, trackers, colors, crop)

    # dataset = "DarkTrack2021"
    # seq = "car_14"
    # main(dataset, seq, trackers, colors, crop)
