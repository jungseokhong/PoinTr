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
import open3d as o3d

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
    partial = np.loadtxt('mug_partial_1.xyz').reshape(1,-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(partial.reshape(-1,3))
    downpcd = pcd.voxel_down_sample(voxel_size=0.0035)
    xyz_load = np.asarray(downpcd.points)
    
    
    ## these are tricks for non-object frame points. you should comment this out for pcd in object frame.
    # means = np.mean(xyz_load, axis=0)
    # new_xyz = (xyz_load - means)
    # print(np.amax(new_xyz), np.amin(new_xyz))
    # if abs(np.amax(new_xyz)) > abs(np.amin(new_xyz)):
    #     new_xyz /= abs(np.amax(new_xyz))
    # else:
    #     new_xyz /= abs(np.amin(new_xyz))
    # idx = new_xyz[:,0]>0
    # np.savetxt("dcd1.xyz", new_xyz[idx])
    # # print(xyz_load.shape)
    # partial = new_xyz[idx].reshape(1,-1,3)

    # print(partial[0,:,0].max(), partial[0,:,0].min(), partial[0,:,1].max(), partial[0,:,1].min(), partial[0,:,2].max(), partial[0,:,2].min())
    ##########
    
    # partial /= 10
    # idx = partial[0,:,0]>0.4
    # np.savetxt("test_bowl_cut_4.xyz", partial[0, idx, :])
    # partial = partial[:,idx,:]
    # print(partial.shape)
    partial = torch.tensor(partial).float()
    # print(partial.shape, partial.dtype)
    # print(partial.size(), partial.double())
    coarse_points, dense_points = test_net(ckpts, config, partial)
    np.savetxt('mug_partial_1_out.xyz', dense_points[0].cpu().numpy())    

if __name__ == '__main__':
    main()
