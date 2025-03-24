import torch
import random
import pickle
import warnings
from opts import *
from os.path import join
from trackers import metrics
from AFLink.AppFreeLink import *
from trackeval.run import evaluate
from AFLink.model import PostLinker
from AFLink.dataset import LinkData



if __name__ == '__main__':
    
    
    # model = PostLinker()
    # model.load_state_dict(torch.load(opt.AFLink_weight_path))
    # dataset = LinkData('', '')

    # # Set path to save
    # save_path = join(opt.eval_path, 'MOT20_val/MOT20-01.txt')
    # out_path = join(opt.eval_path, 'MOT20_aflink/MOT20_val/MOT20-01.txt')
    # os.makedirs(join(opt.eval_path, 'MOT20_aflink/MOT20_val'), exist_ok=True)

    # linker = AFLink(path_in=save_path, path_out=out_path, model=model, dataset=dataset,
    #                             thrT=(opt.thrT_low , opt.thrT_high), thrS=opt.thrS, thrP=opt.thrP)
    # linker.link()
    
    
    setting_dict = {'gt_folder': opt.dataset_root + opt.dataset + '/train',
                    'gt_loc_format': '{gt_folder}/{seq}/gt/gt.txt',
                    'trackers_folder': opt.eval_path, #join(opt.eval_path, 'MOT20_val'),
                    'tracker': opt.dataset + '_' + opt.mode,
                    'dataset': opt.dataset}
    evaluate(setting_dict)
