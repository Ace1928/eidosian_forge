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
def sample_inputs_binary_cross_entropy_with_logits(op_info, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    make_prob = partial(make, low=0, high=1)
    reductions = ('mean', 'sum', 'none')

    def make_weight_shape_kwargs():
        kwargs = []
        for shape in ((1,), (1, S), S, (S, S)):
            kwargs.extend([((S, S), dict(reduction=reduction, weight=make(shape))) for reduction in reductions])
        return kwargs
    shapes_and_kwargs = [*[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))], *[((S, S), dict(reduction=reduction)) for reduction in reductions], *make_weight_shape_kwargs(), *[((S, S), dict(reduction=reduction, pos_weight=make((S,), low=0))) for reduction in reductions], *[((S, S), dict(reduction=reduction, weight=make((S, S)), pos_weight=make((S,), low=0))) for reduction in reductions]]
    for shape, kwargs in shapes_and_kwargs:
        yield SampleInput(make(shape, requires_grad=requires_grad), args=(make_prob(shape, requires_grad=requires_grad),), kwargs=kwargs)