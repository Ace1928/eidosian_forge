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
def sample_inputs_interpolate(mode, self, device, dtype, requires_grad, **kwargs):
    N, C = (2, 3)
    D = 4
    S = 3
    L = 5
    align_corners_options: Tuple[Any, ...] = (None,)
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        align_corners_options = (True, False, None)
    ranks_for_mode = {'nearest': [1, 2, 3], 'nearest-exact': [1, 2, 3], 'linear': [1], 'bilinear': [2], 'bicubic': [2], 'trilinear': [3], 'area': [1, 2, 3]}

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + [size] * rank)
        return tuple([size] * rank)
    if mode in ('bilinear', 'bicubic') and dtype == torch.uint8:
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, high=256 if dtype == torch.uint8 else None)
        rank = 2
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            yield SampleInput(make_arg(shape(270, rank), memory_format=memory_format), shape(130, rank, False), scale_factor=None, mode=mode, align_corners=False)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for align_corners in align_corners_options:
        for rank in ranks_for_mode[mode]:
            yield SampleInput(make_arg(shape(D, rank)), shape(S, rank, False), scale_factor=None, mode=mode, align_corners=align_corners)
            yield SampleInput(make_arg(shape(D, rank)), shape(L, rank, False), scale_factor=None, mode=mode, align_corners=align_corners)
            for recompute_scale_factor in [False, True]:
                for scale_factor in [1.7, 0.6]:
                    yield SampleInput(make_arg(shape(D, rank)), size=None, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)