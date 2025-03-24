import os
import argparse
from os.path import join


class Opts:
    def __init__(self):
        # Initialize
        self.parser = argparse.ArgumentParser()

        # Basic Setting
        self.parser.add_argument('--dataset', default='MOT20', help='MOT17, MOT20')
        self.parser.add_argument('--mode', default='val', help='val or test')
        self.parser.add_argument('--dataset_root', default='/kaggle/working/datasets/')
        self.parser.add_argument('--save_dir', default='./outputs/')

        # For tracking
        self.parser.add_argument('--max_distance', default=0.4, type=float)
        self.parser.add_argument('--max_iou_distance', default=0.85, type=float)
        self.parser.add_argument('--min_len', default=4, type=float)
        self.parser.add_argument('--max_age', default=35, type=float)
        self.parser.add_argument('--ema_beta', default=0.92, type=float) #0.90
        self.parser.add_argument('--gating_lambda', default=0.94, type=float)
        self.parser.add_argument("--min_box_area", default=80, type=float)
        self.parser.add_argument("--conf_thresh", default=0.5, type=float)

        # For Post-processing
        self.parser.add_argument('--AFLink', default=True, action='store_true')
        self.parser.add_argument('--interpolation', default=True, action='store_true')
        self.parser.add_argument('--thrT_low', default=8, type=int)
        self.parser.add_argument('--thrT_high', default=20, type=int)
        self.parser.add_argument('--thrS', default=70, type=int)
        self.parser.add_argument('--thrP', default=0.03, type=float)

        #occlusion parameters
        self.parser.add_argument('--max_iou_threshold', default=0.7, type=float)
        self.parser.add_argument('--proximity_distance', default=70, type=float)
        self.parser.add_argument('--min_occlusion_frames', default=3, type=int)
        self.parser.add_argument('--proximity_ratio_threshold', default=0.3, type=float)
        self.parser.add_argument('--id_switch_penalty', default=1e5, type=float)

        #For Eval
        self.parser.add_argument("--eval_path", default="../TrackOutputs/BoostTrack", type=str)

    def parse(self):
        # Initialize
        opt = self.parser.parse_args()

        # Set directories, paths
        opt.save_dir += '%s_%s' % (opt.dataset, opt.mode)
        opt.dataset_dir = "/kaggle/working/datasets/MOT20/test" if opt.mode == 'test' else "/kaggle/working/datasets/MOT20/train"

        # opt.det_feat_path = '../outputs/2. det_feat/%s_%s.pickle' % (opt.dataset, opt.mode)
        opt.det_feat_path = "/kaggle/working/AdapTrack/outputs/2. det_feat/MOT20_test.pickle"
        # opt.AFLink_weight_path = './AFLink/AFLink_epoch20.pth'
        opt.AFLink_weight_path = './newmodel_epoch20_tmp.pth'

        # Set others
        # opt.conf_thresh = 0.6 if opt.dataset == 'MOT17' else 0.71
        opt.vid_names = os.listdir(opt.dataset_dir)

        # Make dir
        os.makedirs(opt.save_dir, exist_ok=True)

        return opt


# Create option file
opt = Opts().parse()
