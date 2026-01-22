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
def sample_inputs_instance_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: Tuple[Tuple[int], dict] = (((S, S, S), {'momentum': 0.5, 'eps': 0.6}), ((S, S, S), {'momentum': 0.5, 'eps': 0.6, 'use_input_stats': True}), ((3, 2, 4), {'momentum': -1.2}), ((3, 2, 4), {'momentum': 0.0}), ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}), ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}))
    for input_shape, kwargs in cases:
        channels = input_shape[1]
        weight = make_arg(channels)
        bias = make_arg(channels)
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        new_kwargs = {'running_mean': running_mean, 'running_var': running_var, 'weight': weight, 'bias': bias, **kwargs}
        yield SampleInput(make_arg(input_shape), args=(), kwargs=new_kwargs)
    weights = [channels, None]
    biases = [None, None]
    for weight_channels, bias_channels in zip(weights, biases):
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        yield SampleInput(make_arg(input_shape), args=(), kwargs={'running_mean': running_mean, 'running_var': running_var, 'weight': make_arg(weight_channels) if weight_channels is not None else None, 'bias': make_arg(bias_channels) if bias_channels is not None else None})
    yield SampleInput(make_arg((1, 2, 3)), kwargs={})