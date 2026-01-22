import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
def expand_bdim(x, x_bdim):
    if x_bdim is None:
        return x.expand(info.batch_size, *x.shape)
    return x.movedim(x_bdim, 0)