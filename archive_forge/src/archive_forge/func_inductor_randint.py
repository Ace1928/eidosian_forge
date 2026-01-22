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
@register_lowering(inductor_prims.randint, type_promotion_kind=None)
def inductor_randint(low: int, high: int, size: List[int], seed: TensorBox, *, offset: int=0):
    assert not config.fallback_random
    size = [*size]
    dtype = torch.int64
    device = seed.get_device()
    random_pos = ir.FixedLayout(device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        return ops.randint64(seed_loader([]), ops.index_expr(random_pos(index), torch.int32), low, high)
    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=[*size])