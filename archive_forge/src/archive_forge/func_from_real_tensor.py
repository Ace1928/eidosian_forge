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
def from_real_tensor(self, fake_mode, t, make_constant=False, shape_env=None, *, source=None, symbolic_context=None, memoized_only=False):
    if not symbolic_context and (not source) and shape_env:
        if (tracing_context := torch._guards.TracingContext.try_get()):
            if t in tracing_context.tensor_to_context:
                symbolic_context = tracing_context.tensor_to_context[t]
                source = symbolic_context.tensor_source
    maybe_memo = self._get_memo(t)
    if maybe_memo is not None:
        return maybe_memo
    if memoized_only:
        return None
    existing_device = t.device
    if t.is_quantized:
        raise UnsupportedFakeTensorException('quantized nyi in meta tensors')
    if type(t) is torch.nn.Parameter:
        assert not make_constant

    def mk_fake_tensor(make_meta_t):
        with no_dispatch():
            return FakeTensor(fake_mode, make_meta_t(), existing_device, constant=t if make_constant else None)
    out = self.meta_converter(t, shape_env=shape_env, callback=mk_fake_tensor, source=source, symbolic_context=symbolic_context)
    if out is NotImplemented:
        raise UnsupportedFakeTensorException('meta converter nyi')
    if make_constant:
        self.add_constant_storage_mapping(out)
    return out