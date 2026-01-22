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
def sample_inputs_group_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], int, float] = (((1, 6, 3), 2, {'eps': 0.5}), ((2, 6, 3), 2, {'eps': -0.5}), ((1, 3), 1, {'eps': 1e-05}), ((0, 2), 1, {'eps': 1e-05}), ((S, S, S), 1, {'eps': 0.5}))
    for input_shape, num_groups, kwargs in cases:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight_tensor = make_arg(channels)
        bias_tensor = make_arg(channels)
        weights = [weight_tensor, None]
        biases = [bias_tensor, None]
        for weight, bias in itertools.product(weights, biases):
            kwargs = {'weight': weight, 'bias': bias, **kwargs}
            yield SampleInput(make_arg(input_shape), num_groups, **kwargs)
    yield SampleInput(make_arg((1, 2)), args=(1,))