'''
Dat Nguyen
Made modifications to support sparse tensors
'''

"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------

This class is based on the TensorFlow code of PixelCNN++:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
In contrast to that code, we predict mixture weights pi for each channel, i.e., mixture weights are "non-shared".
Also, x_min, x_max and L are parameters, and we implement a function to get the CDF of a channel.

# ------
# Naming
# ------

Note that we use the following names through the code, following the code PixelCNN++:
    - x: targets, e.g., the RGB image for scale 0
    - l: for the output of the network;
      In Fig. 2 in our paper, l is the final output, denoted with p(z^(s-1) | f^(s)), i.e., it contains the parameters
      for the mixture weights.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
import torchvision





# Note that for RGB, we predict the parameters mu, sigma, pi and lambda. Since RGB has C==3 channels, it so happens that
# the total number of channels needed to predict the 4 parameters is 4 * C * K (for K mixtures, see final paragraphs of
# Section 3.4 in the paper). Note that for an input of, e.g., C == 4 channels, we would need 3 * C * K + 6 * K channels
# to predict all parameters. To understand this, see Eq. (7) in the paper, where it can be seen that for \tilde \mu_4,
# we would need 3 lambdas.
# We do not implement this case here, since it would complicate the code unnecessarily.
_NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
_NUM_PARAMS_OTHER = 3  # mu, sigma, pi

_LOG_SCALES_MIN = -7.
_MAX_K_FOR_VIS = 10


CDFOut = namedtuple('CDFOut', ['logit_probs_c_sm',
                               'means_c',
                               'log_scales_c',
                               'K',
                               'targets'])


def non_shared_get_Kp(K, C,rgb):
    """ Get Kp=number of channels to predict. See note where we define _NUM_PARAMS_RGB above """
    if rgb:  # finest scale
        return _NUM_PARAMS_RGB * C * K
    else:
        return _NUM_PARAMS_OTHER * C * K


def non_shared_get_K(Kp, C, rgb):
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    if rgb:
        return Kp // (_NUM_PARAMS_RGB * C)
    else:
        return Kp // (_NUM_PARAMS_OTHER * C)


# --------------------------------------------------------------------------------


class DiscretizedMixLogisticLoss():
    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        # whether to use coefficients lambda to weight means depending on previously outputed means.
        self.use_coeffs = rgb_scale
        # P means number of different variables contained in l, l means output of network
        self._num_params = _NUM_PARAMS_RGB if rgb_scale else _NUM_PARAMS_OTHER

        # NOTE: in contrast to the original code, we use a sigmoid (instead of a tanh)
        # The optimizer seems to not care, but it would probably be more principaled to use a tanh
        # Compare with L55 here: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L55
        self._nonshared_coeffs_act = torch.sigmoid

        # Adapted bounds for our case.
        self.bin_width = (x_max - x_min) / (L-1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
                (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)


    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N*C*H*W)  # NHW -> scalar

    def cdf_step_non_shared(self, l, targets, c_cur, C, x_c=None) -> CDFOut:
        assert c_cur < C

        # NKHW         NKHW     NKHW
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(c_cur, C, l, x_c)

        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)  # NKHW, pi_k
        return CDFOut(logit_probs_c_softmax, means_c, log_scales_c, K, targets.to(l.device))

    def sample(self, l, C):
        return self._non_shared_sample(l, C)

    def get_loss(self, x, l, scale=0):
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, '{},{} not in {},{}'.format(
                x.min(), x.max(), self.x_min, self.x_max)

        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales, K = self._extract_non_shared(x, l)

        # # visualize pi, means, variances
        # self.summarizer.register_images(
        #         'val', {f'dmll/{scale}/c{c}': lambda c=c: _visualize_params(logit_pis, means, log_scales, c)
        #                 for c in range(x.shape[1])})

        centered_x = x - means  # NCKHW

        # Calc P = cdf_delta
        # all of the following is NCKHW
        inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        plus_in = inv_stdv * (centered_x + self.bin_width/2)  # sigma' * (x - mu + 0.5)
        cdf_plus = torch.sigmoid(plus_in)  # S(sigma' * (x - mu + 1/255))
        min_in = inv_stdv * (centered_x - self.bin_width/2)  # sigma' * (x - mu - 1/255)
        cdf_min = torch.sigmoid(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
        # the following two follow from the definition of the logistic distribution
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
        # NCKHW, P^k(c)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min

        # NOTE: the original code has another condition here:
        #   tf.where(cdf_delta > 1e-5,
        #            tf.log(tf.maximum(cdf_delta, 1e-12)),
        #            log_pdf_mid - np.log(127.5)
        #            )
        # which handles the extremly low porbability case. Since this is only there to stabilize training,
        # and we get fine training without it, I decided to drop it
        #
        # so, we have the following if, where I put in the x_upper_bound and x_lower_bound values for RGB
        # if x < 0.001:                         cond_C
        #       log_cdf_plus                    out_C
        # elif x > 254.999:                     cond_B
        #       log_one_minus_cdf_min           out_B
        # else:
        #       log(cdf_delta)                  out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        # NOTE, we adapt the bounds for our case
        cond_B = (x > self.x_upper_bound).float()
        out_B = (cond_B * log_one_minus_cdf_min + (1. - cond_B) * out_A)
        cond_C = (x < self.x_lower_bound).float()
        # NCKHW, =log(P^k(c))
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B

        # combine with pi, NCKHW, (-inf, 0]
        log_probs_weighted = log_probs.add(
                log_softmax(logit_pis, dim=2))  # (-inf, 0]

        # final log(P), NCHW
        return -log_sum_exp(log_probs_weighted, dim=2)  # NCHW

    def get_loss_ycocg(self, x, l, cur_c):
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        # assert x.min() >= self.x_min and x.max() <= self.x_max, '{},{} not in {},{}'.format(
        #         x.min(), x.max(), self.x_min, self.x_max)

        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales, K = self._extract_non_shared(x, l)
        x=x[:,cur_c,None]
        logit_pis=logit_pis[:,cur_c,None]
        means=means[:,cur_c,...]
        log_scales=log_scales[:,cur_c,None]

        centered_x = x - means  # NCKHW

        # Calc P = cdf_delta
        # all of the following is NCKHW
        inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        plus_in = inv_stdv * (centered_x + self.bin_width/2)  # sigma' * (x - mu + 0.5)
        cdf_plus = torch.sigmoid(plus_in)  # S(sigma' * (x - mu + 1/255))
        min_in = inv_stdv * (centered_x - self.bin_width/2)  # sigma' * (x - mu - 1/255)
        cdf_min = torch.sigmoid(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
        # the following two follow from the definition of the logistic distribution
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
        # NCKHW, P^k(c)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min


        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        # NOTE, we adapt the bounds for our case
        cond_B = (x > self.x_upper_bound).float()
        out_B = (cond_B * log_one_minus_cdf_min + (1. - cond_B) * out_A)
        cond_C = (x < self.x_lower_bound).float()
        # NCKHW, =log(P^k(c))
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B

        # combine with pi, NCKHW, (-inf, 0]
        log_probs_weighted = log_probs.add(
                log_softmax(logit_pis, dim=2))  # (-inf, 0]

        # final log(P), NCHW
        return -log_sum_exp(log_probs_weighted, dim=2)  # NCHW

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C, self.rgb_scale)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels, there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW
        means = l[:, 1, ...]  # NCKHW
        log_scales = torch.clamp(l[:, 2, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        x = x.reshape(N, C, 1, H, W)

        if self.use_coeffs:
            assert C == 3  # Coefficients only supported for C==3, see note where we define _NUM_PARAMS_RGB
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])  # NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            means_r, means_g, means_b = means[:, 0, ...], means[:, 1, ...], means[:, 2, ...]  # each NKHW
            coeffs_g_r,  coeffs_b_r, coeffs_b_g = coeffs[:, 0, ...], coeffs[:, 1, ...], coeffs[:, 2, ...]  # each NKHW
            means = torch.stack(
                    (means_r,
                     means_g + coeffs_g_r * x[:, 0, ...],
                     means_b + coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]), dim=1)  # NCKHW again

        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K

    def _extract_non_shared_c(self, c, C, l, x=None):
        """
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        assert c < C, f'{c} >= {C}'

        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self.rgb_scale)

        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs_c = l[:, 0, c, ...]  # NKHW
        means_c = l[:, 1, c, ...]  # NKHW
        log_scales_c = torch.clamp(l[:, 2, c, ...], min=_LOG_SCALES_MIN)  # NKHW, is >= -7

        if self.use_coeffs and c != 0:
            unscaled_coeffs = l[:, 3, ...]  # NCKHW, coeffs_g_r, coeffs_b_r, coeffs_b_g
            if c == 1:
                assert x is not None
                coeffs_g_r = torch.sigmoid(unscaled_coeffs[:, 0, ...])  # NKHW
                means_c += coeffs_g_r * x[:, 0, ...]
            elif c == 2:
                assert x is not None
                coeffs_b_r = torch.sigmoid(unscaled_coeffs[:, 1, ...])  # NKHW
                coeffs_b_g = torch.sigmoid(unscaled_coeffs[:, 2, ...])  # NKHW
                means_c += coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]

        #      NKHW           NKHW     NKHW
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l, C):
        """ sample from model """
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C)
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW

        # sample mixture indicator from softmax
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)  # NCKHW
        sel = torch.argmax(
                logit_probs - torch.log(-torch.log(u)),  # gumbel sampling
                dim=2)  # argmax over K, results in NCHW, specifies for each c: which of the K mixtures to take
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))

        sel = sel.unsqueeze(2)  # NC1HW

        means = torch.gather(l[:, 1, ...], 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(l[:, 2, ...], 2, sel).squeeze(2), min=_LOG_SCALES_MIN)

        # sample from the resulting logistic, which now has essentially 1 mixture component only.
        # We use inverse transform sampling. i.e. X~logistic; generate u ~ Unfirom; x = CDF^-1(u),
        #  where CDF^-1 for the logistic is CDF^-1(y) = \mu + \sigma * log(y / (1-y))
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)  # NCHW
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))  # NCHW

        if self.use_coeffs:
            assert C == 3

            clamp = lambda x_: torch.clamp(x_, 0, 255.)

            # Be careful about coefficients! We need to use the correct selection mask, namely the one for the G and
            #  B channels, as we update the G and B means! Doing torch.gather(l[:, 3, ...], 2, sel) would be completly
            #  wrong.
            coeffs = torch.sigmoid(l[:, 3, ...])
            sel_g, sel_b = sel[:, 1, ...], sel[:, 2, ...]
            coeffs_g_r = torch.gather(coeffs[:, 0, ...], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, 1, ...], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, 2, ...], 1, sel_b).squeeze(1)

            # Note: In theory, we should go step by step over the channels and update means with previously sampled
            # xs. But because of the math above (x = means + ...), we can just update the means here and it's all good.
            x0 = clamp(x[:, 0, ...])
            x1 = clamp(x[:, 1, ...] + coeffs_g_r * x0)
            x2 = clamp(x[:, 2, ...] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)
        return x


def log_prob_from_logits(logit_probs):
    """ numerically stable log_softmax implementation that prevents overflow """
    # logit_probs is NKHW
    m, _ = torch.max(logit_probs, dim=1, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=1, keepdim=True))


# TODO(pytorch): replace with pytorch internal in 1.0, there is a bug in 0.4.1
def log_softmax(logit_probs, dim):
    """ numerically stable log_softmax implementation that prevents overflow """
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))


def log_sum_exp(log_probs, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m, _        = torch.max(log_probs, dim=dim)
    m_keep, _   = torch.max(log_probs, dim=dim, keepdim=True)
    # == m + torch.log(torch.sum(torch.exp(log_probs - m_keep), dim=dim))
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)




def _iter_Kdim_normalized(t, normalize=True):
    """ normalizes t, then iterates over Kdim (1st dimension) """
    K = t.shape[0]

    if normalize:
        lo, hi = float(t.min()), float(t.max())
        t = t.clamp(min=lo, max=hi).add_(-lo).div_(hi - lo + 1e-5)

    for k in range(min(_MAX_K_FOR_VIS, K)):
        yield t[k, ...]  # HW


def _extract_non_shared_c( c, C, l, x=None):
    """
    Same as _extract_non_shared but only for c-th channel, used to get CDF
    """
    assert c < C, f'{c} >= {C}'

    B,Kp,N ,_= l.shape

    _num_params = _NUM_PARAMS_RGB if C==3 else _NUM_PARAMS_OTHER
    use_coeffs=True if C==3 else False
    K = non_shared_get_K(Kp, C, rgb=use_coeffs)

    l = l.reshape(B, _num_params, C, K,N,1) # B3CKN or B4CKN
    logit_probs_c = l[:, 0, c, ...]  # BKN
    means_c = l[:, 1, c, ...]  # BKN
    log_scales_c = torch.clamp(l[:, 2, c, ...], min=_LOG_SCALES_MIN)  # BKN, is >= -7

    if use_coeffs and c != 0:

        unscaled_coeffs = l[:, 3, ...]  # NCKHW, coeffs_g_r, coeffs_b_r, coeffs_b_g
        if c == 1:
            assert x is not None
            coeffs_g_r = torch.sigmoid(unscaled_coeffs[:, 0, ...])  # NKHW
            #print("checking: ", means_c.shape, coeffs_g_r.shape,x.shape, x[:, 0,...].shape)

            means_c += coeffs_g_r * x[:, 0, ...]#.permute(1,0)
        elif c == 2:
            assert x is not None
            coeffs_b_r = torch.sigmoid(unscaled_coeffs[:, 1, ...])  # NKHW
            coeffs_b_g = torch.sigmoid(unscaled_coeffs[:, 2, ...])  # NKHW
            means_c += coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]

    #      BKN           BKN     BKN
    return logit_probs_c, means_c, log_scales_c, K
def compute_metrics(sampled_probs, true_input):
    #sampled_probs = predicts.detach().clone()  # torch.softmax(predicts,dim=1)
    sampled_probs = sampled_probs.cpu()
    true_input =  true_input.long().cpu()

    _, t1 = torch.topk(sampled_probs, 1, 1, True, True)
    correct_t1 = torch.eq(true_input[:, None], t1)
    Top1_accuracy = correct_t1.float().mean()


    _, t3 = torch.topk(sampled_probs, 3, 1, True, True)
    correct_t3 = torch.eq(true_input[:, None], t3).any(dim=1)
    Top3_accuracy = correct_t3.float().mean()

    _, t5 = torch.topk(sampled_probs, 5, 1, True, True)
    correct_t5 = torch.eq(true_input[:, None], t5).any(dim=1)
    Top5_accuracy = correct_t5.float().mean()

    # _, t20 = torch.topk(sampled_probs, 20, 1, True, True)
    # correct_t20 = torch.eq(true_input[:, None], t20).any(dim=1)
    # Top20_accuracy = correct_t20.float().mean()
    #print(Top1_accuracy)
    return Top1_accuracy,Top3_accuracy, Top5_accuracy

def cdf_2_pdf_acc(cdf,sym):
    pdf=cdf[:,:,:,1:]-cdf[:,:,:,:-1]
    b, h, w, c = pdf.shape
    pdf=pdf.reshape((b*h*w,c))
    return compute_metrics(pdf,sym)

def acc_from_mol(data, bn, l, dmll):
    totalbits=0
    B, C, N,_ = bn.shape

    data = data.detach().clone()
    l = l.detach().clone()

    decoded_bn = torch.zeros(B,C,N,1, dtype=torch.float32).to(l.device)

    targets = torch.linspace(dmll.x_min - dmll.bin_width / 2,
                             dmll.x_max + dmll.bin_width / 2,
                             dmll.L + 1, dtype=torch.float32, device=l.device)
    t1=0
    t3=0
    t5=0
    for c_cur in range(C):

        S_c=data[:,c_cur,...].to(torch.int16)
        S_c = S_c.to('cpu', non_blocking=True)
        S_c = S_c.reshape(-1).contiguous()


        #BKN            #BKN    BKN
        logit_probs_c, means_c, log_scales_c, K = _extract_non_shared_c(c_cur, C, l, decoded_bn)
        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)
        decoded_bn[:, c_cur, ...] = bn[:, c_cur]

        targets = targets.to('cpu')
        means_c = means_c.to('cpu')
        logit_probs_c_softmax = logit_probs_c_softmax.to('cpu')
        log_scales_c = log_scales_c.to('cpu')
        if not (targets.is_cuda == means_c.is_cuda == log_scales_c.is_cuda == logit_probs_c_softmax.is_cuda):
            raise ValueError('targets, means, log_scales, logit_probs_softmax must all be on the same device! Got '
                             f'{targets.device}, {means_c.device}, {log_scales_c.device}, {logit_probs_c_softmax.device}.')
        if S_c.is_cuda:
            raise ValueError('sym must be on CPU!')

        #out_bytes = torchdac.encode_logistic_mixture(targets, means_c, log_scales_c, logit_probs_c_softmax, S_c)
        cdf, cdf_float = _get_uint16_cdf(logit_probs_c_softmax, targets, means_c, log_scales_c)
        Top1, Top3, Top5=cdf_2_pdf_acc(cdf_float, S_c)
        #print("Top 1 RGB:", c_cur,Top1)
        t1+=Top1
        t3+=Top3
        t5+=Top5
    return t1/C, t3/C, t5/C

def acc_from_mol_ycocg(data, bn, l, dmlly, dmllc):
    totalbits=0
    B, C, N,_ = bn.shape

    data = data.detach().clone()
    l = l.detach().clone()

    decoded_bn = torch.zeros(B,C,N,1, dtype=torch.float32).to(l.device)


    t1=0
    t3=0
    t5=0
    for c_cur in range(C):

        S_c=data[:,c_cur,...].to(torch.int16)
        S_c = S_c.to('cpu', non_blocking=True)
        S_c = S_c.reshape(-1).contiguous()
        if(c_cur==0):
            targets = torch.linspace(dmlly.x_min - dmlly.bin_width / 2,dmlly.x_max + dmlly.bin_width / 2,dmlly.L + 1, dtype=torch.float32, device=l.device)
        else:
            targets = torch.linspace(dmllc.x_min - dmllc.bin_width / 2, dmllc.x_max + dmllc.bin_width / 2, dmllc.L + 1,
                                     dtype=torch.float32, device=l.device)

        #BKN            #BKN    BKN
        logit_probs_c, means_c, log_scales_c, K = _extract_non_shared_c(c_cur, C, l, decoded_bn)
        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)
        decoded_bn[:, c_cur, ...] = bn[:, c_cur]

        targets = targets.to('cpu')
        means_c = means_c.to('cpu')
        logit_probs_c_softmax = logit_probs_c_softmax.to('cpu')
        log_scales_c = log_scales_c.to('cpu')
        if not (targets.is_cuda == means_c.is_cuda == log_scales_c.is_cuda == logit_probs_c_softmax.is_cuda):
            raise ValueError('targets, means, log_scales, logit_probs_softmax must all be on the same device! Got '
                             f'{targets.device}, {means_c.device}, {log_scales_c.device}, {logit_probs_c_softmax.device}.')
        if S_c.is_cuda:
            raise ValueError('sym must be on CPU!')

        #out_bytes = torchdac.encode_logistic_mixture(targets, means_c, log_scales_c, logit_probs_c_softmax, S_c)
        cdf, cdf_float = _get_uint16_cdf(logit_probs_c_softmax, targets, means_c, log_scales_c)
        Top1, Top3, Top5=cdf_2_pdf_acc(cdf_float, S_c)
        #print("Top 1 RGB:", c_cur,Top1)
        t1+=Top1
        t3+=Top3
        t5+=Top5
    return t1/C, t3/C, t5/C

def _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales):
    # print("Shape: ", logit_probs_softmax.shape, targets.shape, means.shape, log_scales.shape)
    # print("Mean: ", logit_probs_softmax.mean(), targets.mean(), means.mean(), log_scales.mean())
    # print("std: ", logit_probs_softmax.std(), targets.std(), means.std(), log_scales.std())

    cdf_float = _get_C_cur_weighted(logit_probs_softmax, targets, means, log_scales)
    cdf=cdf_float
    #print("cdf float last axis: ",cdf.shape, cdf[:,:,:,-1].mean())
    cdf = _renorm_cast_cdf_(cdf_float, precision=16)

    cdf = cdf.cpu()
    return cdf, cdf_float.cpu()


def _get_C_cur_weighted(logit_probs_softmax_c, targets, means_c, log_scales_c):
    C_cur = _get_C_cur(targets, means_c, log_scales_c)  # NKHWL
    C_cur = C_cur.mul(logit_probs_softmax_c.unsqueeze(-1)).sum(1)  # NHWL
    return C_cur


def _get_C_cur(targets, means_c, log_scales_c):  # NKHWL
    """
    :param targets: Lp floats
    :param means_c: NKHW
    :param log_scales_c: NKHW
    :return:
    """
    #print("cp 3", targets.shape, means_c.shape, log_scales_c.shape)
    # NKHW1
    inv_stdv = torch.exp(-log_scales_c).unsqueeze(-1)
    # NKHWL'
    centered_targets = (targets - means_c.unsqueeze(-1))
    # NKHWL'
    cdf = centered_targets.mul(inv_stdv).sigmoid()  # sigma' * (x - mu)
    return cdf


def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf