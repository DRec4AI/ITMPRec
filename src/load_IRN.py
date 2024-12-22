# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from test_models import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed, data_partition, data_partition2, data_partition3, res_func
import random
from tqdm import tqdm
from simulators_original import RecSim
import copy
from mymodel.influentialRS import InfluentialNet,IRSNN
import torch.nn as nn

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def load_model(config, device="cuda:0"):
    """
    Load existing nn1
    """
    # Load model information
    model_store_path = config.output_dir + config.data_name
    model_info = torch.load(model_store_path + '/irn_params64.pth.tar')

    # Restore the core module
    net = InfluentialNet(config)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(model_info['state_dict'])

    # Restore the handler
    irn = IRSNN(config, net, device)
    irn.optimizer.load_state_dict(model_info['optimizer'])
    return irn

