import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@classmethod
def create_multilayer(cls, device: torch.device, dtype: torch.dtype, inner_fns: Sequence[Callable[..., Any]], ranges: List[Expr], reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
    """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
    reduction_numel = sympy_product(reduction_ranges)
    need_mask = not V.graph.sizevars.is_expr_static_and_true(sympy.Eq(reduction_numel % split, 0))
    if need_mask and reduction_type != 'welford_combine':

        def constant(idx, reduction_idx, value):
            return ops.constant(value, dtype)
        return cls.create_multilayer(device=device, dtype=dtype, inner_fns=(inner_fns[0], partial(constant, value=0), partial(constant, value=1)), ranges=ranges, reduction_ranges=reduction_ranges, reduction_type='welford_combine', split=split, reduction_hint=reduction_hint)
    block_size = FloorDiv(reduction_numel + (split - 1), split)
    intermediates = WelfordReduction.create(device, dtype, tuple((cls._multilayer_wrap_loader(loader, reduction_ranges, reduction_numel, split, block_size, default=0) for loader in inner_fns)), [*ranges, split], [block_size], reduction_type, reduction_hint)
    for i in intermediates:
        i.realize()
    i_loaders = [i.make_loader() for i in intermediates]

    def intermediate_loader_fn(index, reduction_index, loader):
        return loader([*index, *reduction_index])
    numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
    reduction_hint = cls._multilayer_second_step_hint(split, numel_hint, reduction_hint)
    return WelfordReduction.create(device, dtype, tuple((partial(intermediate_loader_fn, loader=i.make_loader()) for i in intermediates)), ranges, [split], 'welford_combine', reduction_hint)