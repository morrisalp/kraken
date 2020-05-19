"""
Virtual Adversarial Training loss
"""
import torch
import numpy as np
import contextlib
import torch.nn.functional as F

from torch import nn

from typing import List, Tuple, Optional, Iterable


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x: torch.Tensor, seq_len: torch.Tensor):
        # ensure softmax
        model.eval()
        with torch.no_grad():
            pred, _ = model(x, seq_len)
        # reenable log_softmax
        model[-1].training = True
        # prepare random unit tensor
        d = torch.rand(x.shape, device=x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                logp_hat, _ = model(x + self.xi * d, seq_len)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            d = torch.sign(d)
            r_adv = d * self.eps

            logp_hat, _ = model(x + r_adv, seq_len)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        model.train()
        return lds
