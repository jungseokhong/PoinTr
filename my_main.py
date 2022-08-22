from tools import test_net
from utils import parser, dist_utils, misc
from utils.config import *
import time
import os
import torch
from my_runner import test_net
import torch.nn as nn
import json
from tools import builder
import numpy as np

def test_net(ckpts, config, partial):
    base_model = builder.model_builder(config.model)
    base_model.cuda()
    base_model.eval()

    # load checkpoints
    builder.load_model(base_model, ckpts)
    
    ## 2048 points
    with torch.no_grad():
        ret = base_model(partial.cuda())
        coarse_points = ret[0]
        dense_points = ret[1]
    return coarse_points, dense_points


def main():
    # config
    ckpts = "./pretrained/pointr_training_from_scratch_c55_best.pth"
    config = cfg_from_yaml_file("./cfgs/ShapeNet55_models/PoinTr.yaml")


    # run
    # partial = torch.from_numpy(np.loadtxt('mug_partial_0.xyz')).view(1,-1,3)
    partial = np.loadtxt('test_bowl.xyz').reshape(1,-1,3)
    print(partial[0,:,0].max(), partial[0,:,0].min(), partial[0,:,1].max(), partial[0,:,1].min(), partial[0,:,2].max(), partial[0,:,2].min())
    # partial /= 10
    idx = partial[0,:,0]>0.4
    np.savetxt("test_bowl_cut_4.xyz", partial[0, idx, :])
    partial = partial[:,idx,:]
    print(partial.shape)
    partial = torch.tensor(partial).float()
    # print(partial.shape, partial.dtype)
    # print(partial.size(), partial.double())
    coarse_points, dense_points = test_net(ckpts, config, partial)
    np.savetxt('test_bowl_out_4.xyz', dense_points[0].cpu().numpy())    

if __name__ == '__main__':
    main()
