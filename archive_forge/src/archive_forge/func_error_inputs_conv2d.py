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
def error_inputs_conv2d(opinfo, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float64)
    make_int_arg = partial(make_tensor, device=device, dtype=torch.int64)
    make_complex_arg = partial(make_tensor, device=device, dtype=torch.complex128)
    yield ErrorInput(SampleInput(make_int_arg((2, 4, 4)), args=(make_int_arg((3, 2, 3, 3)), make_arg((3,)))), error_regex='should be the same')
    yield ErrorInput(SampleInput(make_arg((2, 4, 4)), args=(make_arg((3, 2, 3, 3)), make_complex_arg((3,)))), error_regex='should be the same')
    yield ErrorInput(SampleInput(make_arg((1, 1, 4, 4)), args=(make_arg((1, 2, 2, 3)), make_arg((1,))), kwargs={'stride': (-1,)}), error_regex='non-positive stride is not supported')
    yield ErrorInput(SampleInput(make_arg((1, 1, 4, 3)), args=(make_arg((1, 2, 2, 4)), make_arg((1,))), kwargs={'padding': (-1,)}), error_regex='negative padding is not supported')
    yield ErrorInput(SampleInput(make_arg((1, 1, 4, 2)), args=(make_arg((1, 1, 2, 5)), make_arg((1,))), kwargs={'dilation': (-1,)}), error_regex='dilation should be greater than zero')
    yield ErrorInput(SampleInput(make_arg((1, 1, 4, 3)), args=(make_arg((1, 2, 2)), make_arg((1,))), kwargs={'padding': 'same'}), error_regex='Expected 3-dimensional input for 3-dimensional weight')
    yield ErrorInput(SampleInput(make_arg((2, 2, 4, 3)), args=(make_arg((2, 2, 1, 3)), make_arg((2,))), kwargs={'groups': 3}), error_regex='expected weight to be at least 3 at dimension 0')
    yield ErrorInput(SampleInput(make_arg((2, 2, 4, 3)), args=(make_arg((2, 2, 1, 3)), make_arg((2,))), kwargs={'padding': 'same', 'groups': 3}), error_regex='expected weight to be at least 3 at dimension 0')
    yield ErrorInput(SampleInput(make_arg((2, 2, 4, 5)), args=(make_arg((2, 2, 1, 4)), make_arg((2,))), kwargs={'padding': 'same', 'groups': -1}), error_regex='non-positive groups is not supported')
    yield ErrorInput(SampleInput(make_arg((2, 2, 4, 3)), args=(make_arg((2, 2, 4, 3)), make_arg((2,))), kwargs={'padding': 'same', 'groups': 0}), error_regex='non-positive groups is not supported')