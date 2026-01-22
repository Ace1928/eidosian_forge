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
def sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
    tensor_nd = partial(make_tensor, (S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_1d = partial(make_tensor, (S,), device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(tensor_nd())
    yield SampleInput(tensor_nd(), dim=1)
    yield SampleInput(tensor_nd(), dim=1, unbiased=True, keepdim=True)
    yield SampleInput(tensor_1d(), dim=0, unbiased=True, keepdim=True)
    yield SampleInput(tensor_1d(), dim=0, unbiased=False, keepdim=False)
    yield SampleInput(tensor_nd(), dim=(1,), correction=1.3)
    yield SampleInput(tensor_nd(), dim=(1,), correction=S // 2)
    yield SampleInput(tensor_nd(), dim=None, correction=0, keepdim=True)
    yield SampleInput(tensor_nd(), dim=None, correction=None)
    yield SampleInput(tensor_nd(), correction=0, keepdim=True)
    yield SampleInput(make_tensor(3, 4, 5, device=device, dtype=dtype, requires_grad=requires_grad), dim=-3)