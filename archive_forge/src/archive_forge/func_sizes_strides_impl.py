import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def sizes_strides_impl(self, sizes, strides):
    op = getattr(sys.modules[__name__], method)
    if sym_function_mode():
        return to_node(self, handle_sym_dispatch(op, ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]), {}))
    size_exprs = [s.expr for s in sizes]
    stride_exprs = [s.expr for s in strides]
    try:
        out = func(size_exprs, stride_exprs)
    except Exception:
        log.warning('failed to eval %s(%s, %s)', method, size_exprs, stride_exprs)
        raise
    size_hints = []
    out_hint = None
    for s in sizes:
        if s.hint is None:
            break
        size_hints.append(s.hint)
    else:
        stride_hints = []
        for s in strides:
            if s.hint is None:
                break
            stride_hints.append(s.hint)
        else:
            out_hint = op(size_hints, stride_hints)
    pytype: Type
    if method.endswith('_indicator'):
        pytype = int
    else:
        pytype = bool
    return SymNode(out, self.shape_env, pytype, out_hint)