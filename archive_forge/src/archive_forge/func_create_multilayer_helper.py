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
def create_multilayer_helper(cls, device: torch.device, dst_dtype: torch.dtype, src_dtype: torch.dtype, wrapper_fn: Callable[..., Any], original_ranges: List[Expr], original_reduction_ranges: List[Expr], new_ranges: List[Expr], new_reduction_ranges: List[Expr], reduction_type: str, split: int, reduction_hint: ReductionHint):
    """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
    intermediate_dtype = dst_dtype if dst_dtype not in (torch.float16, torch.bfloat16) else torch.float
    intermediate = Reduction.create(device, intermediate_dtype, src_dtype, wrapper_fn, new_ranges, new_reduction_ranges, reduction_type, reduction_hint)
    intermediate.realize()
    intermediate_loader = intermediate.make_loader()

    def intermediate_fn(index, reduction_index):
        return intermediate_loader([*index, *reduction_index])
    numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
    reduction_hint = cls._multilayer_second_step_hint(split, numel_hint, reduction_hint)
    assert original_ranges == new_ranges[:len(original_ranges)]
    return TensorBox.create(Reduction(device, dst_dtype, intermediate_fn, original_ranges, new_ranges[len(original_ranges):], reduction_type, src_dtype, reduction_hint))