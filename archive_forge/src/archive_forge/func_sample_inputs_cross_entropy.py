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
def sample_inputs_cross_entropy(op_info, device, dtype, requires_grad, **kwargs):
    batch_size, num_classes = shape = (2, 3)
    reductions = ('mean', 'sum', 'none')
    input_shape_and_kwargs: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = [(shape, {}), ((*shape, 1), {}), ((*shape, 1, 2), {}), ((*shape, 1, 2, 3), {}), *[(shape, dict(reduction=reduction)) for reduction in reductions], *[(shape, dict(weight=make_tensor((num_classes,), device=device, dtype=dtype), reduction=reduction)) for reduction in reductions], (shape, dict(ignore_index=1))]
    for (input_shape, kwargs), probabilities_target in itertools.product(input_shape_and_kwargs, (False, True)):
        input = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=requires_grad)
        if probabilities_target:
            if 'ignore_index' in kwargs:
                continue
            target = make_tensor(input_shape, low=0, high=1, device=device, dtype=dtype, requires_grad=requires_grad)
        else:
            target = make_tensor((batch_size, *input_shape[2:]), low=0, high=num_classes, device=device, dtype=torch.long)
            if 'ignore_index' in kwargs and torch.all(target == kwargs['ignore_index']):
                target[0] = random.sample(sorted(set(range(num_classes)) - {kwargs['ignore_index']}), 1)[0]
        yield SampleInput(input, target, **kwargs)