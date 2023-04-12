from glob import glob
import random as rn
import os
import sys

import numpy as np
from pyntcloud import PyntCloud
import torch
from torch.utils.data import Dataset, Subset
import MinkowskiEngine as ME
import math as m
from torchvision.transforms import Compose

# torch.manual_seed(1)
# np.random.seed(1)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class PCdataset(Dataset):
  def __init__(self, files,block_size, transforms=None):
    self.files=np.asarray(files)
    self.transforms=transforms
    self.bz=block_size
  def __len__(self):
    return len(self.files)
  def __getitem__(self, idx):
    pc=PyntCloud.from_file(self.files[idx])
    try:
        cols=['x', 'y', 'z','red', 'green', 'blue']
        points=pc.points[cols].values
        points=points.astype(np.float32)

    except:
        cols = ['x', 'y', 'z', 'r', 'g', 'b']
        points = pc.points[cols].values
        points = points.astype(np.float32)

    if(self.transforms):
        points=self.transforms(points)

    _, indices = np.unique(points[:,:3], axis=0, return_index=True)
    points=points[indices]
    xyz=points[:,:3]
    feats=points[:,3:]
    feats=feats.astype(np.float32)
    occups=np.ones(shape=(xyz.shape[0],1),dtype=np.float32)
    return xyz, feats,occups



def data_collector(training_dirs,valid_dirs,transform,subset,filterout_noisy_data_rate, params):
    total_train = []

    for training_dir in training_dirs:

        training_dir = training_dir + '**/**.ply'


        files = glob(training_dir, recursive=True)

        total_len=len(files)
        #only select some largest files
        sizes = [os.stat(x).st_size for x in files]
        files_with_sizes = list(zip(files, sizes))
        files_sorted_by_points = sorted(files_with_sizes, key=lambda x: -x[1])
        files_sorted_by_points = files_sorted_by_points[:int(len(files) * filterout_noisy_data_rate)] #filtered out some last sparse portion
        files = list(zip(*files_sorted_by_points))
        files = list(files[0])
        total_train = np.concatenate((total_train, files), axis=0)
        print('Selected ', len(files), ' in total ', total_len, ' in ', training_dir)

    assert len(total_train) > 0
    training_files=total_train[:int(len(total_train)*subset)]

    rn.shuffle(training_files)  # shuffle file

    train_set_size = int(training_files.shape[0] * 0.8)
    #valid_set_size = len(training_files) - train_set_size

    total_train=training_files[:train_set_size]
    total_valid=training_files[train_set_size:]

    if(transform==3):
        rotation = Rotation(64)
        sampling = Random_sampling()
        da = Compose([rotation, sampling])
        train_set = PCdataset(total_train, 64,8, da)
        valid_set = PCdataset(total_valid, 64,8, transforms=None)
    elif (transform == 2):
        sampling = Random_sampling()
        da = Compose([sampling, ])
        train_set = PCdataset(total_train, 64, da)
        valid_set = PCdataset(total_valid, 64, transforms=None)
    elif(transform==1):
        rotation=Rotation(64)
        da = Compose([rotation, ])
        train_set = PCdataset(total_train, 64, da)
        valid_set = PCdataset(total_valid, 64, transforms=None)
    elif (transform == 4):
        changeorder = ChangeOrder("grb")
        da = Compose([changeorder, ])
        train_set = PCdataset(total_train, 64, da)
        valid_set = PCdataset(total_valid, 64, da)
    elif (transform == 5):
        colortransform = RGBtoYUV()
        da = Compose([colortransform, ])
        train_set = PCdataset(total_train, 64,8, da)
        valid_set = PCdataset(total_valid, 64, 8,da)
    elif (transform == 6):
        changeorder = ChangeOrder("grb")
        sampling = Random_sampling()
        train_da = Compose([changeorder, sampling])
        valid_da = Compose([changeorder])
        train_set = PCdataset(total_train, 64,8, train_da)
        valid_set = PCdataset(total_valid, 64,8, valid_da)
    elif (transform == 7): #ycocg

        color_tf = RGBtoYCgCo(9)
        train_da = Compose([color_tf])
        valid_da = Compose([color_tf])
        train_set = PCdataset(total_train, 64, train_da)
        valid_set = PCdataset(total_valid, 64,valid_da)
    elif (transform == 8): #ycocg with sampling
        sampling = Random_sampling()
        color_tf = RGBtoYCgCo(9)
        train_da = Compose([color_tf,sampling])
        valid_da = Compose([color_tf])
        train_set = PCdataset(total_train, 64, train_da)
        valid_set = PCdataset(total_valid, 64, valid_da)
    elif (transform == 9):  #cgcoy
        sampling = Random_sampling()
        color_tf = RGBtoCgCoY(9)
        train_da = Compose([color_tf, sampling])
        valid_da = Compose([color_tf])
        train_set = PCdataset(total_train, 64, train_da)
        valid_set = PCdataset(total_valid, 64, valid_da)
    elif(transform==0):
        sampling = Random_sampling()
        norm = normailize(8)
        train_da = Compose([sampling, norm])
        valid_da = Compose([norm])
        train_set = PCdataset(total_train, 64,train_da)
        valid_set = PCdataset(total_valid, 64,valid_da)
    elif (transform == 12):
        sampling = Random_sampling()
        norm = normailize(8)
        changeorder = ChangeOrder("gbr")
        train_da = Compose([sampling, norm, changeorder, ])
        test_da= Compose([ norm, changeorder, ])
        train_set = PCdataset(total_train, 64, train_da)
        valid_set = PCdataset(total_valid, 64, test_da)
    elif (transform == 11):
        sampling = Random_sampling()
        norm = normailize2(8)
        da = Compose([sampling, norm])
        train_set = PCdataset(total_train, 64, da)
        valid_set = PCdataset(total_valid, 64, da)
    elif (transform == 10):
        sampling = Random_sampling()
        norm = normailize_128(8)
        da = Compose([sampling, norm])
        train_set = PCdataset(total_train, 64, da)
        valid_set = PCdataset(total_valid, 64, da)



    print('Total blocks for training: ', len(total_train))
    print('Total blocks for validation: ', len(total_valid))


    training_generator = torch.utils.data.DataLoader(train_set,collate_fn=ME.utils.SparseCollation(), **params)
    params['shuffle']=False
    valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn=ME.utils.SparseCollation(),**params)
    return training_generator, valid_generator


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])


class Rotation(object):  # randomly rotate a point cloud

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, points):
        # print('before',points.shape[0])
        da_degree = np.random.randint(1, 45)
        theta=np.random.choice([0,da_degree], p=[0.5, 0.5])
        rotmtx = [Rx, Ry, Rz]
        R = rotmtx[np.random.randint(0, 2)](theta)

        coords=points[:,:3]
        coords = coords * R
        coords = coords - np.min(coords, axis=0)
        coords = np.round(coords)

        points[:,:3]=coords
        # print('larger than block: ',np.count_nonzero(points>=self.block_size))
        points = np.delete(points, np.where(np.max(points[:,:3], axis=1) >= self.block_size)[0], 0)
        points = np.delete(points, np.where(np.min(points[:,:3], axis=1) < 0)[0], 0)
        # print('after',points.shape[0])

        return points

class ChangeOrder(object):

    def __init__(self, order):
        self.order = order

    def __call__(self, points):
        if(self.order=="grb"):
            points[:, [4, 3]] = points[:, [3, 4]] # rgb to bgr
        elif (self.order=="gbr"):
            points[:, [4, 3]] = points[:, [3, 4]] #rgb to grb
            points[:, [5,4]] = points[:, [4,5]] #grb to gbr

        return points

class RGBtoYUV(object):



    def __call__(self, points):
        color=points[:,3:]
        color=RGB2YUV(color)
        points[:,3:]=color
        return points
def RGB2YUV(rgb): # ITU-R BT.709 standard
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[ :, 1:] += 128.0
    yuv=np.round(yuv)
    return yuv


# input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
# output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:,  0] -= 179.45477266423404
    rgb[:, 1] += 135.45870971679688
    rgb[:, 2] -= 226.8183044444304
    return rgb

class Random_sampling(object):  # randomly remove points from pcs
    def __call__(self, points):
        rates = [0, 0.2, 0.4, 0.6]
        rate = np.random.choice(rates, p=[0.5, 0.16, 0.17, 0.17])
        idx = np.random.choice(np.arange(0,points.shape[0]),replace=False, size=int(points.shape[0] * (1 - rate)))
        points = points[idx, :]

        return points

class RGBtoYCgCo(object):  # randomly remove points from pcs
    def __init__(self, chroma_bitdepth):
        self.chromabitdepth=chroma_bitdepth
    def __call__(self, points):

        points=np.round(points)
        color = points[:, 3:].astype(np.int16)
        color=transformRGBToYCgCoR(self.chromabitdepth-1,color)
        color = color.astype(np.float32)
        color[:, 0] += 127.5

        step=(pow(2,self.chromabitdepth)-1.)/2.
        color=(color-step)/step
        points[:, 3:] = color
        return points

class RGBtoCgCoY(object):  # randomly remove points from pcs
    def __init__(self, chroma_bitdepth):
        self.chromabitdepth=chroma_bitdepth
    def __call__(self, points):

        points=np.round(points)
        color = points[:, 3:].astype(np.int16)
        color = transformRGBToCgCoY(self.chromabitdepth-1,color)
        color = color.astype(np.float32)
        color[:, 2] += 127.5

        step=(pow(2,self.chromabitdepth)-1.)/2.
        color=(color-step)/step
        points[:, 3:] = color
        return points

class normailize(object):  # normalize color to -1, 1
    def __init__(self, bitdepth):
        self.bitdepth=bitdepth
    def __call__(self, points):
        points=np.round(points)
        color = points[:, 3:]
        step=(pow(2,self.bitdepth)-1)/2
        color=(color-step)/step
        points[:, 3:] = color

        return points

class normailize2(object):  # normalize color to -1, 1
    def __init__(self, bitdepth):
        self.bitdepth=bitdepth
    def __call__(self, points):
        points=np.round(points)
        color = points[:, 3:]
        step=(pow(2,self.bitdepth)-1)/2
        color=color-step
        points[:, 3:] = color
        return points

class normailize_128(object):  # normalize color to -1, 1
    def __init__(self, bitdepth):
        self.bitdepth=bitdepth
    def __call__(self, points):
        points=np.round(points)
        color = points[:, 3:]
        step=(pow(2,self.bitdepth)-1)/2
        color=(color-step)
        points[:, 3:] = color

        return points
def transformRGBToYCgCoR(bitdepth, rgb):
    g = rgb[:,1]
    b = rgb[:, 2]
    r = rgb[:, 0]
    co = r - b
    t = b + (co >> 1) # chia doi
    cg = g - t
    y = t + (cg >> 1) # chia doi

    offset = 1 << bitdepth # 2^bitdepth
    return np.column_stack((y,  co + offset,cg + offset,))


def transformRGBToCgCoY(bitdepth, rgb):
    r = rgb[:, 0]
    g = rgb[:,1]
    b = rgb[:, 2]

    co = r - b
    t = b + (co >> 1) # chia doi
    cg = g - t
    y = t + (cg >> 1) # chia doi

    offset = 1 << bitdepth # 2^bitdepth
    return np.column_stack((cg + offset, co + offset,y))

def transformYCgCoRToRGB( bitDepth,  ycgco):

    offset = 1 << bitDepth
    y0 = ycgco[:,0]
    cg = ycgco[:,1] - offset
    co = ycgco[:,2] - offset

    t = y0 - (cg >> 1)

    g = cg + t
    b = t - (co >> 1)
    r = co + b

    maxVal = (1 << bitDepth) - 1
    g = np.clip(g, 0, maxVal)
    b = np.clip(b, 0, maxVal)
    r = np.clip(r, 0, maxVal)

    return np.column_stack((r,g,b))