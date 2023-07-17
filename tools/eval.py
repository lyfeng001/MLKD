from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

sys.path.append("./")
import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from toolkit.datasets import *
from toolkit.visualization import draw_success_precision

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,default='UAVDark135',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--dataset_dir', default='',type=str, help='dataset root directory')
parser.add_argument('--tracker_result_dir',default='', type=str, help='tracker result root')
parser.add_argument('--vis',dest='vis', default=True,action='store_true')

parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    ##       student_spa     stu_1_checkpoint_e2

    tracker_dir = '/home/user/V4R/LYF/pysot-master2/test_result/student_mcl'#os.path.join(args.tracker_path, args.dataset) student_spa stu_1_checkpoint_e2
    # trackers = glob(os.path.join(args.tracker_path,
    #                               args.dataset,
    #                               args.tracker_prefix))
    trackers = ['0.15_0.423_0.30']#[x.split('/')[-1] for x in args.tracker_prefix.split(',')]

    root = '/home/user/V4R/LYF/pysot-master2/testing_dataset/UAVDark135/UAVDark135'#args.dataset_dir + args.dataset

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'UAVDark135' in args.dataset:
        dataset = UAVDARKDataset(args.dataset, root)
    elif 'DarkTrack2021' in args.dataset:
        dataset = DARKTRACKDataset(args.dataset, root)



    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    # ret = benchmark.eval_success(trackers)
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=18):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=18):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
            show_video_level=args.show_video_level)
    if args.vis:
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)

if __name__ == '__main__':
    main()
