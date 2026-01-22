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
def sample_inputs_fractional_max_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 3, 9, 9), 3), ((1, 3, 9, 9), (4, 4)), ((1, 3, 9, 9), (6, 6)), ((2, 3, 9, 9), (3, 3)), ((1, 1, 4, 4), (2, 2)), ((1, 2, 6, 6), (4, 4)))
    for input_shape, kernel_size in cases:
        for return_indices in [False, True]:
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=2, return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=(2, 3), return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_ratio=(0.5, 0.5), return_indices=return_indices)