import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def int_exp(self, x_int, scaling_factor):
    with torch.no_grad():
        x0_int = torch.floor(self.x0 / scaling_factor)
    x_int = torch.max(x_int, self.const * x0_int)
    q = floor_ste.apply(x_int / x0_int)
    r = x_int - x0_int * q
    exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
    exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.const - q)), min=0)
    scaling_factor = exp_scaling_factor / 2 ** self.const
    return (exp_int, scaling_factor)