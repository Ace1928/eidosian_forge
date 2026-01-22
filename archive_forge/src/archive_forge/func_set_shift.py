import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def set_shift(self, y_int):
    with torch.no_grad():
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        shift = torch.log2(torch.sqrt(var_int / 2 ** self.max_bit)).ceil().max()
        shift_old = self.shift
        self.shift = torch.max(self.shift, shift)
        logger.info(f'Dynamic shift adjustment: {int(shift_old)} -> {int(self.shift)}')