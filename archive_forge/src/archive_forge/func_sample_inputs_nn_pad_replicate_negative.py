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
def sample_inputs_nn_pad_replicate_negative(op_info, device, dtype, requires_grad, **kwargs):
    cases: tuple = (((5, 3, 4, 4), (-4, 5, 0, 0)), ((6, 2, 4, 4), (0, 0, 2, -4)), ((5, 6, 4, 4), (5, -4, -4, 3)), ((4, 2, 5, 5), (-2, -1, 4, 6)), ((2, 6, 5, 5), (8, -1, -1, -3)), ((8, 1, 5, 5), (-2, -1, -1, -3)))
    make_inp = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for shape, pad in cases:
        yield SampleInput(make_inp(shape), args=(pad, 'replicate'))