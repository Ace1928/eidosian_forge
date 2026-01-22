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
def sample_inputs_upsample(mode, self, device, dtype, requires_grad, **kwargs):
    N, C = (2, 3)
    D = 4
    S = 3
    L = 5
    ranks_for_mode = {'nearest': [1, 2, 3], 'bilinear': [2]}

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return torch.Size([N, C] + [size] * rank)
        return torch.Size([size] * rank)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for rank in ranks_for_mode[mode]:
        yield SampleInput(make_arg(shape(D, rank)), size=shape(S, rank, False))
        yield SampleInput(make_arg(shape(D, rank)), size=shape(L, rank, False))
        yield SampleInput(make_arg(shape(D, rank)), scale_factor=1.7)
        yield SampleInput(make_arg(shape(D, rank)), scale_factor=0.6)