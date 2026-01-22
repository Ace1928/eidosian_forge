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
def sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: Tuple[Tuple[int], dict] = (((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}), ((3, 2, 4), {'training': False, 'momentum': -1.2}), ((3, 1), {'training': True, 'momentum': 0.0}), ((0,), {'training': True}), ((0,), {'training': False}), ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}), ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}), ((2, 1), {}))
    for input_shape, kwargs in cases:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        yield SampleInput(make_arg(input_shape), args=(running_mean, running_var, weight, bias), kwargs=kwargs)
    weights = [channels, None, None]
    biases = [None, channels, None]
    is_training = [True, False, False]
    for weight, bias, training in zip(weights, biases, is_training):
        yield SampleInput(make_arg(input_shape), args=(running_mean, running_var, make_arg(channels), make_arg(channels)), kwargs={'training': training})
    yield SampleInput(make_arg((1, 2, 3)), args=(None, None, None, None), kwargs={'training': True})