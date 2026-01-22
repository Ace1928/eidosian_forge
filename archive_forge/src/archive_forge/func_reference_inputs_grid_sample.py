from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def reference_inputs_grid_sample(op_info, device, dtype, requires_grad, **kwargs):
    batch_size = 2
    num_channels = 3
    height = 345
    width = 456
    modes = ('bilinear', 'nearest', 'bicubic')
    align_cornerss = (False, True)
    padding_modes = ('zeros', 'border', 'reflection')
    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = (torch.cos(a), torch.sin(a))
    s1, s2 = (1.23, 1.34)
    theta = torch.tensor([[[ca / s1, sa, 0.0], [-sa, ca / s2, 0.0]]], dtype=dtype, device=device)
    theta = theta.expand(batch_size, 2, 3).contiguous()
    x = torch.arange(batch_size * num_channels * height * width, device=device)
    x = x.reshape(batch_size, num_channels, height, width).to(torch.uint8)
    x = x.to(dtype=dtype)
    x.requires_grad_(requires_grad)
    for mode, padding_mode, align_corners in itertools.product(modes, padding_modes, align_cornerss):
        grid = torch.nn.functional.affine_grid(theta, size=(batch_size, num_channels, height, width), align_corners=align_corners)
        yield SampleInput(x, grid, mode, padding_mode, align_corners)