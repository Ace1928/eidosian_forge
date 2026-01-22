from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def make_combine_region(scan_op):
    in_scalar_tys = [t.type.scalar for t in input]
    prototype = function_type(in_scalar_tys, in_scalar_tys * 2)
    region = scan_op.get_region(0)
    with _insertion_guard(_builder):
        param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
        block = _builder.create_block_with_parent(region, param_types)
        args = [tensor(block.arg(i), ty) for i, ty in enumerate(prototype.param_types)]
        results = _generator.call_JitFunction(combine_fn, args, kwargs={})
        if isinstance(results, tensor):
            handles = [results.handle]
        else:
            handles = [r.handle for r in results]
        _builder.create_scan_ret(*handles)