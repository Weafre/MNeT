import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as F
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

# QUANTIZATION CONFIGS
SIGMA = 2
LEVELS= [-1,1]
STEPS = 25
'''
creating MNeT neural network
'''
class residualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.no_filter = h
        self.spconva = ME.MinkowskiConvolution(in_channels= h, out_channels=h, kernel_size=3, dimension=3)
        self.spconvb = ME.MinkowskiConvolution( in_channels=h, out_channels=h, kernel_size=3, dimension=3)
        self.resnetblock = nn.Sequential(
            self.spconva,
            ME.MinkowskiReLU(),
            self.spconvb,
        )

    def forward(self, x):
        identity = x
        out = self.resnetblock(x)
        out += identity
        return out



class SoftQuantizer(nn.Module):
    # SoftQuantizer class adapted from
    # https://github.com/fab-jul/L3C-PyTorch/blob/master/src/modules/quantizer.py

    def __init__(self,levels,sigma=0.2):
        super(SoftQuantizer, self).__init__()
        assert levels.dim() == 1, 'Expected 1D levels, got {}'.format(levels)
        self.levels = levels#.to('cuda:0')
        self.sigma = sigma
        self.L = self.levels.size()[0]

    def __repr__(self):
        return '{}(sigma={})'.format(
            self._get_name(), self.sigma)

    def forward(self, x):
        """
        :param x: NF
        :return:x_soft, x_hard , symbols
        """
        self.levels=self.levels.to(x.device)
        # print('x',x)
        feat = x.features
        assert feat.dim() == 2, 'Expected NF, got {}'.format(feat.size())
        N, C = feat.shape
        # make x into M1, where M=N*C
        feat = feat.view(N*C, 1)
        # ML, d[..., l] gives distance to l-th level
        d = torch.pow(feat - self.levels, 2)
        # ML, \sum_l d[..., l] sums to 1
        phi_soft = F.softmax(-self.sigma * d, dim=-1)
        # - Calcualte soft assignements ---
        # M1, soft assign x to levels
        feat_soft = torch.sum(self.levels * phi_soft, dim=-1)

        feat_soft = feat_soft.view(N, C)
        # - Calcualte hard assignements ---
        # M1, symbols_hard[..., i] contains index of symbol to use
        _, symbols_hard = torch.min(d.detach(), dim=-1)
        # NC
        symbols_hard = symbols_hard.view(N, C)
        # NC, contains value of symbol to use
        feat_hard = self.levels[symbols_hard]


        feat_soft.data = feat_hard  # assign data, keep gradient


        x_soft = ME.SparseTensor(features=feat_soft, coordinate_manager=x.coordinate_manager,
                                 coordinate_map_key=x.coordinate_map_key)
        x_hard = ME.SparseTensor(features=feat_hard, coordinate_manager=x.coordinate_manager,
                                 coordinate_map_key=x.coordinate_map_key)


        return x_soft, x_hard, symbols_hard



class encoder_block_light(nn.Module):
    def __init__(self, input_channel,output_channel,no_filter,  no_res):
        super(encoder_block_light, self).__init__()
        self.no_res=no_res
        self.conv1=nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=no_filter, kernel_size=5, stride=1,
                                    dimension=3),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
            )

        self.res_bl=nn.Sequential(
            *[residualBlock(no_filter) for _ in range(no_res)],
        )

        self.to_lower_conv=nn.Sequential(
            ME.MinkowskiConvolution(in_channels=no_filter, out_channels=no_filter, kernel_size=3, stride=1,dimension=3),
            ME.MinkowskiELU(),
        )

        self.to_quantize_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=no_filter, out_channels=output_channel, kernel_size=1,stride=1, dimension=3),
            )

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.res_bl(x1)
        x1 = ME.SparseTensor(features=x1.F, coordinate_manager=x2.coordinate_manager,
                             coordinate_map_key=x2.coordinate_map_key)
        x3=x2+x1
        to_lower=self.to_lower_conv(x3)
        to_quantize=self.to_quantize_conv(x3)
        return to_lower,to_quantize


class decoder_block_light(nn.Module):
    def __init__(self, input_channel, output_channel, no_res,no_fil,*args, **kwargs):
        super(decoder_block_light, self).__init__()
        self.no_res=no_res
        self.no_mid_filter=no_fil
        self.conv1=nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=self.no_mid_filter, kernel_size=1, stride=1,
                                    dimension=3),)

        self.res_bl=nn.Sequential(
            *[residualBlock(self.no_mid_filter) for _ in range(no_res)],
        )
        self.upsampler=nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=self.no_mid_filter, out_channels=self.no_mid_filter, kernel_size=3, stride=2,
                                    dimension=3),
        )
        self.to_upper_conv=nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.no_mid_filter, out_channels=self.no_mid_filter, kernel_size=3, stride=1,dimension=3),
            ME.MinkowskiELU(),)

        self.to_prob_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.no_mid_filter, out_channels=self.no_mid_filter, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiConvolution(in_channels=self.no_mid_filter, out_channels=output_channel, kernel_size=1, stride=1, dimension=3), )



    def forward(self, x,f,ref_coord):
        x1=self.conv1(x)

        if(f!=None):
            f = ME.SparseTensor(features=f.F, coordinate_manager=x1.coordinate_manager,
                                 coordinate_map_key=x1.coordinate_map_key)
            x1=x1+f

        x2 = self.res_bl(x1)
        x3=x1+x2

        x3=self.upsampler(x3)
        x3=self.to_upper_conv(x3)
        probs=self.to_prob_conv(x3)
        return x3,probs

class ms_net_light(nn.Module):
    def __init__(self, no_res,steps,noltfil,noscale,nomix, training=True,quantize=False):
        super(ms_net_light, self).__init__()
        filter=64
        enc_in_features=[3,filter,filter]
        self.steps = steps
        self.noltfil = noltfil

        dec_out_features = [4*3*nomix, 3*noltfil*nomix, 3*noltfil*nomix]
        dec_no_fil=[filter,filter,filter]

        self.noscale=noscale
        self.istraining=training
        self.enc=nn.ModuleList([encoder_block_light(input_channel=enc_in_features[i],output_channel=noltfil,no_filter=filter, no_res=no_res) for i in range(noscale)])
        if quantize:
            self.q_levels = nn.Parameter(torch.linspace(LEVELS[0],LEVELS[1],self.steps), requires_grad=False)
            self.quantizer=[SoftQuantizer(self.q_levels,sigma=SIGMA) for _ in range(self.noscale)]

        self.dec=nn.ModuleList([decoder_block_light(input_channel=self.noltfil,output_channel=dec_out_features[j], no_res=no_res,no_fil=dec_no_fil[j]) for j in range(noscale)])
    def forward(self,x):
        #ENCODER
        en_out_list=[x]
        q_in_list=[]
        for i in range(self.noscale):
            e_out,q_in=self.enc[i](en_out_list[-1])
            q_in_list.append(q_in)
            en_out_list.append(e_out)


        #QUANTIZATION
        q_soft_list=[]
        q_hard_list=[x]
        sym_out=[x.F]
        if hasattr(self, 'quantizer'):
            for i in range(self.noscale):
                q_soft,q_hard,sym=self.quantizer[i](q_in_list[i])
                q_soft_list.append(q_soft)
                q_hard_list.append(q_hard)
                sym_out.append(sym)

        else:
            for i in range(self.noscale):
                q_soft_list.append(q_in_list[i])
                q_hard_list.append(q_in_list[i])

        # DECODER
        f_list=[None]
        p_list=[torch.ones(q_soft_list[-1].F.shape)] # uniform distribution on the lowest scale
        bn=[]
        if(self.istraining): #use q soft in training
            for i in range(self.noscale)[::-1]: #decoding in the reverse order
                f,p=self.dec[i](q_soft_list[i],f_list[-1],q_hard_list[i]) #using e_out_list just for coordinate reference
                f_list.append(f)
                p_list.append(p.F)
                bn.append(q_hard_list[i+1].F)
        else: # use q hard as decoder input in bitencoding
            for i in range(self.noscale)[::-1]: #decoding in the reverse order
                f,p=self.dec[i](q_hard_list[i+1],f_list[-1],q_hard_list[i]) #using e_out_list just for coordinate reference
                f_list.append(f)
                p_list.append(p.F)
                bn.append(q_hard_list[i + 1].F)
        bn.append(x.F)

        return bn,p_list,sym_out[::-1]

if __name__ == "__main__":
    net=ms_net(5)
    device = torch.device("cuda")
    net=net.to(device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

