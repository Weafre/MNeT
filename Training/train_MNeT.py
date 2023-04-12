import argparse
import numpy as np
import pathlib
import sys

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from DataPreprocessing.training_data_pipeline2 import data_collector
from Training.model_MNeT import  ms_net_light
from utils.mol_criterion import  DiscretizedMixLogisticLoss, acc_from_mol


working_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(working_dir))

'''
pytorch lightning trainer module
'''
def train_ms_net(args):
    params = {'batch_size': args.batch,
              'shuffle': True,
              'num_workers': 4}


    training_generator, valid_generator = data_collector(args.trainset, args.validset, args.useDA,args.subset, 0.99, params)
    #seed = 42
    step=5
    gm=0.75
    #torch.manual_seed(seed)

    num_devices = min(args.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")


    try:
        pl_module = ms_module.load_from_checkpoint(
            checkpoint_path=args.input_model_path, no_res=args.nores,
            no_bins=args.nobins, noltfil=args.noltfil, lr=args.lr,
            lrstep=step, gm=gm, noscale=args.noscale,nomix=args.nomix, training=True,
            quantize=args.quantize, flag=args.flag)
        print("loaded model")
    except:
        pl_module = ms_module(no_res=args.nores, no_bins=args.nobins,
                              noltfil=args.noltfil, lr=args.lr, lrstep=step,
                              gm=gm, noscale=args.noscale,nomix=args.nomix, training=True,
                              quantize=args.quantize, flag=args.flag)

    #pl_module.load_model(args.input_model_path)
    #pl_module = ms_module.load_from_checkpoint(args.input_model_path, strict=False,no_res=args.nores,dim=args.dim, lr=args.lr, lrstep=step, gm=gm)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath= args.saving_model_path + args.flag +'/'+ str(args.nobins)+'_'+str(args.noltfil)+'_'+str(args.noscale)+'/',
        filename=  "best_val_checkpoint_model_"+"_lr_"+str(args.lr)+"_b_"+ str(args.batch)+"_da_"+ str(args.useDA)+"_nores_"+str(args.nores)+ "_schedule_"+str(step)+str(gm)+"_nobins_"+str(args.nobins)+"_noltfil_"+str(args.noltfil)+"-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(args.saving_model_path + args.flag +'/'+ str(args.nobins)+'_'+str(args.noltfil)+'_'+str(args.noscale)+'/log/')
    trainer = Trainer(auto_lr_find=True,max_epochs=-1, gpus=num_devices, strategy="ddp",callbacks=[checkpoint_callback, early_stop_callback],logger=tb_logger,accumulate_grad_batches=args.bacc)
    trainer.fit(pl_module, training_generator, valid_generator)



class ms_module(LightningModule):
    def __init__(self,no_res, no_bins,noltfil, lr,lrstep,gm,noscale,nomix,training=True,quantize=False,flag=None):
        super().__init__()
        self.model=ms_net_light(no_res,no_bins,noltfil,noscale,nomix,training,quantize)
        self.no_bins=no_bins-1
        self.lossfc=nn.CrossEntropyLoss()
        self.step=lrstep
        self.noltfil=noltfil
        self.gm=gm
        self.lr=lr * 1e-5
        self.train_loss=0.
        self.valid_loss=0.
        self.valid_loss_min=np.Inf
        self.T1=0
        self.noscale=noscale
        self.flag=flag
        self.loss_dmol_rgb = DiscretizedMixLogisticLoss(rgb_scale=True, x_min=0, x_max=255, L=256)
        self.loss_dmol_n = DiscretizedMixLogisticLoss(rgb_scale=False, x_min=-1, x_max=1., L=self.no_bins+1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        coords, feats, occups = batch
        sparse_inputs = ME.SparseTensor(feats, coords)
        predicts=self.model(sparse_inputs)
        losses=[]
        for cur_s in range(self.noscale + 1):
            bn = predicts[0][cur_s]  # nxc
            prediction = predicts[1][cur_s]  # n x l --> 1 x l x n
            symout = predicts[2][cur_s]  # n x c --> 1 x c x n

            rgb = cur_s == self.noscale
            loss = self.loss_dmol_rgb if rgb else self.loss_dmol_n
            # bottleneck and symbol out of rgb layer is differ than other layers
            symout = torch.round((symout * 127.5 + 127.5)) if rgb else symout
            bn = symout if rgb else bn
            bn = bn.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # nxc-->1 x c x n --> 1 x c x n x 1 pretend to be BCHW
            prediction = prediction.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x l --> 1 x l x n
            symout = symout.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x c --> 1 x c x n
            if (cur_s!=0):
                losses.append(loss.get_loss(bn, prediction, self.noscale - cur_s).sum())
                if(batch_idx%100==0):
                    T1, T3,T5=acc_from_mol(symout, bn,prediction, loss)
                    self.log('T1_'+str(cur_s), T1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    self.log('T5_'+str(cur_s), T5, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.T1 = self.T1 + ((1 / (batch_idx + 1)) * (T1 - self.T1))
            else:
                B,C,H,W = symout.shape
                L = self.no_bins+1
                losses.append(B*C*H*W* np.log(L))
        loss = sum(losses) /( coords.shape[0]*np.log(2))


        self.log("loss", loss,on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.train_loss =  self.train_loss + ((1 / (batch_idx + 1)) * (loss.item() - self.train_loss))

        return loss

    def validation_step(self, batch, batch_idx):
        coords, feats, occups = batch
        sparse_inputs = ME.SparseTensor(feats, coords)


        predicts = self.model(sparse_inputs)
        losses = []
        for cur_s in range(self.noscale + 1):
            bn = predicts[0][cur_s]  # nxc
            prediction = predicts[1][cur_s]  # n x l --> 1 x l x n
            symout = predicts[2][cur_s]  # n x c --> 1 x c x n

            rgb = cur_s == self.noscale
            loss = self.loss_dmol_rgb if rgb else self.loss_dmol_n
            # bottleneck and symbol out of rgb layer is differ than other layers
            symout = torch.round((symout * 127.5 + 127.5)) if rgb else symout
            bn = symout if rgb else bn

            bn = bn.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # nxc-->1 x c x n --> 1 x c x n x 1 pretend to be BCHW
            # print(prediction.shape)
            prediction = prediction.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x l --> 1 x l x n
            symout = symout.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # n x c --> 1 x c x n

            if (cur_s != 0):
                losses.append(loss.get_loss(bn, prediction, self.noscale - cur_s).sum())
                if (batch_idx % 100 == 0):
                    T1, T3, T5 = acc_from_mol(symout, bn, prediction, loss)
                    self.log('vT1_' + str(cur_s), T1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.log('vT5_' + str(cur_s), T5, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                    self.T1 = self.T1 + ((1 / (batch_idx + 1)) * (T1 - self.T1))
            else:
                B, C, H, W = symout.shape
                L = self.no_bins + 1
                losses.append(B * C * H * W * np.log(L))

        loss = sum(losses) /(np.log(2)* coords.shape[0])

        self.valid_loss = self.valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - self.valid_loss))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr , betas=(0.95, 0.995))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=self.gm)
        return [optimizer], [scheduler]

    def validation_epoch_end(self,output):
        self.log("val_loss",self.valid_loss,on_epoch=True, prog_bar=True, logger=True)
        if (self.valid_loss <= self.valid_loss_min):
            self.valid_loss_min = self.valid_loss


        self.valid_loss=0.
        self.train_loss = 0.
        self.T1=0
    def training_epoch_end(self,output):
        self.log("train_loss", self.train_loss,on_epoch=True, prog_bar=False, logger=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='color voxeldnn training')
    parser.add_argument("-blocksize", type=int, default=64, help="input block size")
    parser.add_argument("-batch", type=int, default=32, help="batch size")
    parser.add_argument("-opt", type=int, default=1, help="Optimizer selection")
    parser.add_argument("-nomix", type=int, default=10, help="number of mixture")
    parser.add_argument("-nores", type=int, default=8, help="number oof residual bloock")
    parser.add_argument("-noscale", type=int, default=3, help="number of scale")

    parser.add_argument("-lr", type=int, default=50, help="actual lr=lr*1e-4")
    parser.add_argument("-subset", type=float, default=1.0, help="subset portion for mock training")

    parser.add_argument("-useDA", '--useDA', type=int,
                        default=0,
                        help='0: no da, 1: only rotation, 2: only subsampling, 3: both')

    parser.add_argument("-noltfil", type=int, default=5, help="number of filter in the latent space")
    parser.add_argument("-color", '--color', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='color training or occupancy training')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='using data augmentation or not')
    parser.add_argument('--quantize', action='store_true', default=False,
                        help='apply quantization or not')
    parser.add_argument("-nobins", type=int, default=51, help="number of bins in quanntizer")

    parser.add_argument("-flag", '--flag', type=str, default='test', help='training flag ')
    parser.add_argument("-trainset", '--trainset', action='append', type=str, help='path to train set ')
    parser.add_argument("-validset", '--validset', action='append', type=str, help='path to valid set ')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-inputmodel", '--input_model_path', type=str, help='path to input model file')
    parser.add_argument("-ngpus", type=int, default=1, help="num_gpus")
    parser.add_argument("-bacc", type=int, default=1, help="gradient accumulation4")
    args = parser.parse_args()

    print("Starting Color training....")
    train_ms_net(args)
    #module_test()
    # python3 -m Training.train_MNeT -trainset ExampleTrainingSet/TrainSet/  -validset ExampleTrainingSet/ValidSet  -flag train_2510 -outputmodel Model/  -nores 8  -useDA 0    --quantize   -lr 50 -noscale 3  -noltfil 5 -nobins 16  -ngpus 1  -batch 2   -bacc 1

