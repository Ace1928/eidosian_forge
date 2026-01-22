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
def sample_inputs_gaussian_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_var = partial(make_tensor, low=0.1, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape(shape):
        yield shape
        yield (*shape[:-1], 1)
        yield shape[:-1]

    def gen_shape_kwargs():
        for s, r in _generate_sample_shape_reduction():
            for t_s, v_s in product(gen_shape(s), gen_shape(s)):
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(full=True, reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(eps=random.uniform(1e-06, 0.001), reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(full=True, eps=random.uniform(1e-06, 0.001), reduction=r))
    for input, target, var, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target, var), kwargs=kwargs)