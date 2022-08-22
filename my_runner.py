import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import numpy as np

def test_net(ckpts, config):
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, ckpts)
    
    ## 2048 points
    ret = base_model(partial)
    coarse_points = ret[0]
    dense_points = ret[1]
    return coarse_points, dense_points
