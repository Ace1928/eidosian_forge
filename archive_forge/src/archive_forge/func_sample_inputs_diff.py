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
def sample_inputs_diff(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases = (((1,), 0, None, None), ((S,), 0, None, None), ((S, 1), 0, None, None), ((S, 1), 1, None, None), ((S, S), 0, None, None), ((S, S), 1, None, None), ((S, S), 0, (1, S), (2, S)), ((S, S), 0, None, (2, S)), ((XS, XS, XS), 1, None, None), ((XS, XS, XS), 2, None, None), ((XS, XS, XS), 1, (XS, 1, XS), (XS, 1, XS)), ((XS, XS, XS), 2, (XS, XS, 1), (XS, XS, 1)), ((XS, XS, XS), 2, (XS, XS, XS), (XS, XS, XS)))
    sample_inputs = []
    for size, dim, size_prepend, size_append in test_cases:
        prepend_size = 0 if size_prepend is None else size_prepend[dim]
        append_size = 0 if size_append is None else size_append[dim]
        dim_size = size[dim] + prepend_size + append_size
        for n in range(dim_size):
            input_tensor = make_arg(size)
            prepend = make_arg(size_prepend) if size_prepend else None
            append = make_arg(size_append) if size_append else None
            yield SampleInput(input_tensor, n, dim, prepend, append)
    yield SampleInput(make_arg((XS, XS, XS)), S + 1, 1)
    yield SampleInput(make_arg((XS, XS, XS)), S * 3 + 2, 2, make_arg((XS, XS, XS)), make_arg((XS, XS, XS)))