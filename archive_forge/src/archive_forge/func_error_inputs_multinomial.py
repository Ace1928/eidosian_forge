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
def error_inputs_multinomial(op_info, device, **kwargs):
    x = torch.empty(1, 2, 3, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(2,)), error_regex='prob_dist must be 1 or 2 dim')
    x = torch.empty(1, 2, dtype=torch.long, device=device)
    yield ErrorInput(SampleInput(x, args=(2,)), error_regex='multinomial only supports floating-point dtypes for input')
    x = torch.empty(1, 2, dtype=torch.double, device=device)
    y = torch.empty(1, 2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(2,), kwargs=dict(out=y)), error_regex='multinomial expects Long tensor out')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(0,)), error_regex='cannot sample n_sample <= 0 samples')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(-1,)), error_regex='cannot sample n_sample <= 0 samples')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(3, False)), error_regex='cannot sample n_sample > prob_dist')
    x = torch.empty(16777217, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(3,)), error_regex='number of categories cannot exceed')
    inputs = ((1.0, -1.0, 1.0), (1.0, inf, 1.0), (1.0, -inf, 1.0), (1.0, 1.0, nan))
    err_msg1 = 'probability tensor contains either `inf`, `nan` or element < 0'
    err_msg2 = 'invalid multinomial distribution'
    rep_arg = (False, True) if torch.device(device).type == 'cpu' else (False,)
    for rep in rep_arg:
        kwargs = {'num_samples': 2, 'replacement': rep}
        for shape in inputs:
            yield ErrorInput(SampleInput(torch.tensor(shape), kwargs=kwargs), error_regex=err_msg1 if rep is False else err_msg2)
        x = torch.zeros(3, device=device)
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)
        x = torch.zeros(3, 3, device=device)
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)
        x[1, :] = 1
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)