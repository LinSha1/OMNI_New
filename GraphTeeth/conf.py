import argparse
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os
import argparse
import pprint
import numpy as np
import torch
import random
import logging
import shutil
import yaml
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset_path', default="../data/OMNI_COCO", type=str, help="experiment dataset")
parser.add_argument('--dataset', default="tooth", type=str, help="experiment dataset tooth")

# Param
parser.add_argument('-b','--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-cl','--num-classes', default=10, type=int, metavar='N', help='num_classes')
parser.add_argument('-lr', '--learning-rate', default=2e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-lr_decay_type', default='cos', type=str, choices=['cos', 'step', 'cyclic'], help='learning rate decay type')
parser.add_argument('-optimizer_type', default='AdamW', type=str, choices=['Adam', 'AdamW', 'SGD'], help='optimizer type')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer-eps', default=1e-8, type=float)
parser.add_argument('--evaluate', action='store_true', help='evaluation mode')
parser.add_argument('--num_proposals', default=50, type=int, metavar='N', help='faster-rcnn proposals boxes num')

# Network and Loss
parser.add_argument('--arc', default='resnet50', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help="backbone architecture resnet")
parser.add_argument('--metric', default="dots", type=str, choices=['dots', 'cosine', 'l1'], help="metric for graph top-K nearest neighbors selection")
parser.add_argument('--lam', default=0.001, type=float, help="lambda for adjusting loss")

# Device and Seed
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

# Experiment
parser.add_argument('--exp-name', default="edge", type=str, help="experiment name for saving checkpoints")


parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')


# ------------------------------


def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg
# ------------------------------

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message = message + '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message = message + '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message = message + '----------------- End -------------------'
    return message


def get_config():

    # args from argparser
    cfg = parser2dict()
    if cfg.dataset == 'tooth':
        with open('config/omni_config.yaml', 'r') as f:
            datasets_cfg = yaml.load(f, Loader=yaml.FullLoader)
            datasets_cfg = edict(datasets_cfg)

    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    cfg.update(datasets_cfg)
    return cfg


def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids


def set_outdir(conf):
    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name,timestr)
    else:
        outdir = os.path.join(default_outdir,conf.exp_name)
        prefix = 'bs_'+str(conf.batch_size)+'_seed_'+str(conf.seed)+'_lr_'+str(conf.learning_rate)
        outdir = os.path.join(outdir,prefix)
    ensure_dir(outdir)
    conf['outdir'] = outdir
    return conf


# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))


def set_logger(cfg):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    if 'loglevel' in cfg:
        loglevel = eval('logging.'+cfg.loglevel)
    else:
        loglevel = logging.INFO


    if cfg.evaluate:
        outname = 'test.log'
    else:
        outname = 'train.log'

    outdir = cfg['outdir']
    log_path = os.path.join(outdir,outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))


def get_path():
    root = r"D:\MScPro\OMNI_New\data\OMNI_COCO\testdata\train"
    annFile = r"D:\MScPro\OMNI_New\data\OMNI_COCO\testdata\annotations\instances_train.json"
    return root,annFile
