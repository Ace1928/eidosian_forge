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
def sample_inputs_nn_unfold(op_info, device, dtype, requires_grad, **kwargs):
    shapes = ((0, 1, 5, 5), (2, 3, 5, 5))
    kernel_sizes = (2, (2, 2), (2, 3))
    dilations = (1, 2, (1, 2))
    paddings = (0, 1, (1, 2))
    strides = (1, 2, (1, 2))
    cases = product(shapes, kernel_sizes, dilations, paddings, strides)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for shape, kernel_size, dilation, padding, stride in cases:
        tensor = make_arg(shape)
        yield SampleInput(tensor, kernel_size, dilation, padding, stride)
    yield SampleInput(make_arg((1, 1, 5, 5)), (3, 3))