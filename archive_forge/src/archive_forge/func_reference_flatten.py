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
def reference_flatten(input, start_dim=0, end_dim=-1):
    in_shape = input.shape
    in_rank = len(in_shape)
    for d in (start_dim, end_dim):
        if not (in_rank == 0 and d in (-1, 0) or -in_rank <= d < in_rank):
            raise IndexError(f'Dimension out of range (expected to be in range of [{-in_rank}, {in_rank - 1}], but got {d}')
    end_dim = end_dim if end_dim >= 0 else in_rank + end_dim
    start_dim = start_dim if start_dim >= 0 else in_rank + start_dim
    if in_rank == 0:
        end_dim = start_dim
    if end_dim < start_dim:
        raise RuntimeError('flatten() has invalid args: start_dim cannot come after end_dim')
    flatten_bit_dim = functools.reduce(operator.mul, in_shape[start_dim:end_dim + 1], 1)
    out_shape = in_shape[:start_dim] + (flatten_bit_dim,) + in_shape[end_dim + 1:]
    return np.reshape(input, out_shape)