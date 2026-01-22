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
def sample_inputs_kl_div(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, low=0.0, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_log(shape):
        out = torch.nn.functional.log_softmax(make_arg(shape), -1)
        out.requires_grad_(requires_grad)
        return out

    def make_prob(shape):
        out = torch.nn.functional.softmax(make_arg(shape), -1)
        out.requires_grad_(requires_grad)
        return out
    shapes = ((2,), (2, 3))
    reductions = ('none', 'mean', 'batchmean', 'sum')
    for shape, reduction, log_target in product(shapes, reductions, (True, False)):
        input = make_log(shape)
        target = make_log(shape) if log_target else make_prob(shape)
        yield SampleInput(input, args=(target,), kwargs=dict(reduction=reduction, log_target=log_target))