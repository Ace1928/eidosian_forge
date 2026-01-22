import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def get_cubic_upsample_coefficients(t):
    A = -0.75
    _1 = ops.constant(1.0, torch.float32)
    c0 = cubic_convolution2(ops.add(t, _1), A)
    c1 = cubic_convolution1(t, A)
    x2 = ops.sub(_1, t)
    c2 = cubic_convolution1(x2, A)
    c3 = cubic_convolution2(ops.add(x2, _1), A)
    return (c0, c1, c2, c3)