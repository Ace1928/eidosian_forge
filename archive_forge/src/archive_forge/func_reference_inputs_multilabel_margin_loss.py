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
def reference_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_target_tensor = partial(torch.tensor, device=device, dtype=torch.long, requires_grad=False)
    inputs = (([], make_target([], low=-1, high=1)), ([S], make_target([S], low=-1, high=S)), ([M, S], make_target([M, S], low=-1, high=S)), ([], make_target_tensor(-1)), ([7], make_target_tensor([2, 0, 6, -1, 4, -1, 6])), ([4, 5], make_target_tensor([[4, -1, 0, -1, 2], [0, 0, 4, 1, 4], [-1, 3, -1, 1, 0], [4, 3, 2, 1, 0]])))
    reductions = (None, 'none', 'mean', 'sum')
    for (shape, target), reduction in product(inputs, reductions):
        kwargs = {}
        if reduction is not None:
            kwargs['reduction'] = reduction
        yield SampleInput(_make_tensor(shape), args=(target,), kwargs=kwargs)