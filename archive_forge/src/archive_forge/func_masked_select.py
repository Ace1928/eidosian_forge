import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
@register_op_impl(lambda func: func is torch.ops.aten.masked_select.default)
def masked_select(fake_mode, func, self, mask):
    if fake_mode.shape_env is None or not fake_mode.shape_env.allow_dynamic_output_shape_ops:
        raise DynamicOutputShapeException(func)
    nnz = fake_mode.shape_env.create_unbacked_symint()
    maxval = sys.maxsize - 1
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size, has_free_symbols
    if not has_free_symbols(arg.numel()):
        if arg.numel() >= 2:
            maxval = int(arg.numel())
    _constrain_range_for_size(nnz, max=maxval)
    return self.new_empty((nnz,))