
import numpy as np

import time
from glob import glob

import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
import pandas as pd
import torch
import os
from utils.octree_partition import partition_octree

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def get_bin_stream_blocks(path_to_ply, pc_level, departition_level):
    # co 10 level --> binstr of 10 level, blocks size =1
    level = int(departition_level)
    pc = PyntCloud.from_file(path_to_ply)
    points = pc.points.values

    color = points[:,3:].astype(np.float32)
    step = (pow(2, 8) - 1.) / 2.
    color = (color - step) / step
    points[:, 3:] = color


    no_oc_voxels = len(points)
    box = int(2 ** pc_level)
    blocks2, binstr2 = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return no_oc_voxels, blocks2, binstr2

def occupancy_map_explore(ply_path, pc_level, departition_level):
    no_oc_voxels, blocks, binstr = get_bin_stream_blocks(ply_path, pc_level, departition_level)
    return blocks, binstr


def pc_partitioning(ply_path, pc_level, departition_level, order="RGB"):
    level = int(departition_level)
    pc = PyntCloud.from_file(ply_path)
    points = pc.points.values


    if(order=="GRB"):
        points[:, [4, 3]] = points[:, [3, 4]]
        print("order changed")
    #points=rgbtoyuv(points)

    box = int(2 ** pc_level)
    blocks, binstr = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return blocks, binstr



def pmf_to_cdf(pmf):

  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0