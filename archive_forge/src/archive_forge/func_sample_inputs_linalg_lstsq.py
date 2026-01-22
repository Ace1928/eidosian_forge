import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix
    device = torch.device(device)
    drivers: Tuple[str, ...]
    if device.type == 'cuda':
        drivers = ('gels',)
    else:
        drivers = ('gels', 'gelsy', 'gelss', 'gelsd')
    deltas: Tuple[int, ...]
    if device.type == 'cpu' or has_cusolver():
        deltas = (-1, 0, +1)
    else:
        deltas = (0,)
    for batch, driver, delta in product(((), (3,), (3, 3)), drivers, deltas):
        shape = batch + (3 + delta, 3)
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        a.requires_grad_(requires_grad)
        b = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(a, b, driver=driver)