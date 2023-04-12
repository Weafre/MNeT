
import os
import argparse
import time
import gzip
import pickle
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F

from torchdac import torchdac
from Training.train_MNeT import ms_module
from  utils.inout import occupancy_map_explore
from utils.mol_criterion import DiscretizedMixLogisticLoss
from Encoder.metadat_codec import save_compressed_file
from utils.inout import  pmf_to_cdf


print('Finished importing')
'''
    Encoding a point cloud's attribute losslessly using MNeT multiscale model
    MNeT predict the probabilities of each color channel on each point in the form of Mixture of Logistics
    Number of output on each layer:
    RGB         layer 1          layer 2
    [4*3*nomix, 3*noltfil*nomix, 3*noltfil*nomix]
    We then need to extract/sample discrete probabilities for each color value (0-255)
    Original MoL processing functions are from https://github.com/fab-jul/L3C-PyTorch
    
'''
def ColorVoxelDNN(args):
    global bbbits, model, nobins, noltfil, device,noscale, acc, sample_op, molrgb, moln

    bbbits=0
    pc_level, ply_path,output_path, modelpath, nobins, noltfil,noscale,signaling, usecuda= args
    departition_level = pc_level - 10
    sequence_name = os.path.split(ply_path)[1]
    sequence=os.path.splitext(sequence_name)[0]

    output_path = output_path+str(sequence)+'/'+signaling
    os.makedirs(output_path,exist_ok=True)

    color_bin = output_path+'.color.bin'
    metadata_file = output_path + '.metadata.bin'
    info = output_path +'.info.pkl'

    molrgb = DiscretizedMixLogisticLoss(rgb_scale=True, x_min=0, x_max=255, L=256)
    moln = DiscretizedMixLogisticLoss(rgb_scale=False, x_min=-1, x_max=1., L=nobins )
    start = time.time()
    boxes, binstr=occupancy_map_explore(ply_path,pc_level,departition_level)
    device = torch.device("cuda" if usecuda else "cpu")

    model = ms_module.load_from_checkpoint(modelpath[0], lr=1, lrstep=1, gm=1,no_res=8,no_bins=nobins,noltfil=noltfil,noscale=noscale,nomix=10, training=False,quantize=True)
    model.eval()
    model.freeze()
    model.to(device)
    print("model loaded")


    #encoding function
    flags=[]

    print("Encoding: ",len(boxes), ' blocks')
    with  open(color_bin , 'wb') as colorbit:
        bp_scale,acc, bpp, ocv = encoding_executer(boxes, colorbit)

    #saving encoding statistic
    with open(info,'wb') as f:
        pickle.dump([acc, bpp, ocv],f)
    #saving metadata
    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)

    color_size= int(os.stat(color_bin).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8 + len(flags)*2+len(boxes)*36

    avg_bpov = (color_size + metadata_size) / np.sum(ocv)

    print('\n \nEncoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)
    print('Models: ',modelpath)
    print('Occupied Voxels: %04d' % np.sum(ocv))
    print('Color bitstream: ', color_bin)
    print('Metadata bitstream', metadata_file )
    print('Encoding information: ', info)
    print('Metadata : ', metadata_size)
    print("Bit per scale: ", 100*np.asarray(bp_scale)/(np.sum(bp_scale)))
    print('Average color bits per occupied voxels: %.04f' % avg_bpov)


def encoding_executer(boxes, color_bits):
    bp_scale=[0,0,0,0]
    acc=np.zeros((len(boxes)))
    bpp=np.zeros((len(boxes)))
    ocv = np.zeros((len(boxes)))
    losses = np.zeros((len(boxes)))
    for i in range(len(boxes)):
        bp_scale, T1_3, br, loss=encode_color_block(boxes[i], bp_scale,color_bits)
        ocv[i]=boxes[i].shape[0]
        losses[i]=loss
    return bp_scale,acc, bpp, ocv



def encode_color_block(block,bps,bitstream=None):
    global  model, device, pdf_info,predicted_label, nobins, noltfil,noscale,acc, molrgb, moln
    coord=block[:,:3]
    feats=block[:,3:].astype(np.float32)

    coords, feats = ME.utils.sparse_collate([coord, ], [feats,])
    sparse_inputs = ME.SparseTensor(feats, coords, device=device)
    predicts = model(sparse_inputs)
    blockbits=0

    for cur_s in range(noscale + 1):
        bn = predicts[0][cur_s] #nxc
        prediction = predicts[1][cur_s] #n x l --> 1 x l x n
        symout = predicts[2][cur_s]#n x c --> 1 x c x n

        rgb = cur_s==noscale
        dll=molrgb if rgb else moln
        # bottleneck and symbol out of rgb layer is differ than other layers
        symout=torch.round((symout * 127.5 + 127.5)) if rgb else symout
        bn = symout if rgb else bn
        bn = bn.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # nxc-->1 x c x n --> 1 x c x n x 1 pretend to be BCHW
        prediction = prediction.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x l --> 1 x l x n
        symout = symout.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x c --> 1 x c x n

        # for sampling on top layer, uncomment this
        '''
         if(cur_s==3):
            sample=dll.sample(prediction,3)
            sample=sample.squeeze().permute(1,0).detach().cpu().numpy()
            print('output: ',sample.shape)
            save_sampled_pcs(coords[:,1:],sample,"Output/red_and_black_sampled_from_dist.ply")

        '''

        nobits=coder(symout, bn,prediction, dll,cur_s,bitstream )
        blockbits += nobits
        bps[cur_s] += nobits
    return bps, 0, blockbits, 0

def coder(data, bn, l, dmll,scale, bitstream):
    totalbits=0
    B, C, N,_ = bn.shape
    targets = torch.linspace(dmll.x_min - dmll.bin_width / 2,
                             dmll.x_max + dmll.bin_width / 2,
                             dmll.L + 1, dtype=torch.float32, device=l.device)
    if(scale!=0):
        _, logit_probs, means, log_scales, K = dmll._extract_non_shared(bn, l)

    for c_cur in range(C):

        S_c=data[:,c_cur,...].to(torch.int16)#.to(l.device)
        S_c = S_c.to('cpu', non_blocking=True)
        S_c = S_c.reshape(-1).contiguous()

        if scale!=0:
            logit_probs_c = logit_probs[:, c_cur, :].contiguous()
            means_c = means[:, c_cur, :].contiguous()
            log_scales_c = log_scales[:, c_cur, :].contiguous()
            logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1).contiguous()
            out_bytes = torchdac.encode_logistic_mixture(targets, means_c, log_scales_c,
                                                                    logit_probs_c_softmax, S_c)
            if (bitstream != None):
                bitstream.write(out_bytes)
            totalbits += len(out_bytes) * 8
        else:
            B, C, N, _ = data.shape
            histo = torch.ones(dmll.L, dtype=torch.float32) / dmll.L
            extendor = torch.ones(B, N, 1 ,dmll.L)
            pr = extendor * histo
            cdf=_get_cdf_from_pr(pr)
            out_bytes = torchdac.encode_cdf(cdf, S_c)
            if (bitstream != None):
                bitstream.write(out_bytes)
            totalbits += len(out_bytes) * 8
    return totalbits

def save_sampled_pcs(coords,att,path ):
    points=np.concatenate((coords,att), axis=1)
    final_target_path = path

    cols=['x', 'y', 'z', 'red', 'green', 'blue']
    points = pd.DataFrame(data=points, columns=cols)
    pc = PyntCloud(points)
    pc.points = pc.points.astype(
        {"x": np.float32, "y": np.float32, "z": np.float32, "red": np.uint8, "green": np.uint8, "blue": np.uint8})
    pc.to_file(final_target_path)


def _get_cdf_from_pr(pr):
    """
    :param pr: NHWL
    :return: NHW(L+1) as int16 on CPU!
    """
    N, H, W, _ = pr.shape
    precision = 16
    cdf = torch.cumsum(pr, -1)
    cdf = cdf.mul_(2**precision)
    cdf = cdf.round()
    cdf = torch.cat((torch.zeros((N, H, W, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=-1)
    cdf = cdf.to('cpu', dtype=torch.int16, non_blocking=True)
    return cdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')
    parser.add_argument("-depth", '--partitioningdepth', type=int,
                        default=3,
                        help='max depth to partition block')
    parser.add_argument("-nobins", '--nobins', type=int,
                        default=5,
                        help='no quantization step in bottle neck')
    parser.add_argument("-noscale", '--noscale', type=int,
                        default=3,
                        help='number of scales')

    parser.add_argument("-nofil", '--nofil', type=int,
                        default=25,
                        help='no of filter in the latent space')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-output", '--outputpath', type=str, help='path to output files')

    parser.add_argument("-model", '--modelpath',action='append', type=str, help='path to input color model  64 .h5 file')
    parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='using cuda or not')
    args = parser.parse_args()
    ColorVoxelDNN([args.octreedepth, args.plypath, args.outputpath, args.modelpath,args.nobins, args.nofil,args.noscale, args.signaling, args.usecuda])
#encoding command
# python3 -m Encoder.MNeT_Encoder -level 10 -ply  ../Datasets/TestPCs/10bits/ricardo_0010.ply -output Output/ -model Model/best_val_checkpoint_model__lr_50_b_32_da_0_nores_8_schedule_50.75_nobins_26_noltfil_5-epoch=125-val_loss=10.42.ckpt -signaling 2510_pooling -nobins 26 -nofil 5 -noscale 3

