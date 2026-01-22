from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def to_ir(self, builder: ir.builder):
    ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
    ret_types = [ret_type.to_ir(builder) for ret_type in self.ret_types]
    return builder.get_function_ty(ir_param_types, ret_types)