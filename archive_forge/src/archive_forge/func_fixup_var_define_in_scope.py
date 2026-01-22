import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def fixup_var_define_in_scope(blocks):
    """Fixes the mapping of ir.Block to ensure all referenced ir.Var are
    defined in every scope used by the function. Such that looking up a variable
    from any scope in this function will not fail.

    Note: This is a workaround. Ideally, all the blocks should refer to the
    same ir.Scope, but that property is not maintained by all the passes.
    """
    used_var = {}
    for blk in blocks.values():
        scope = blk.scope
        for inst in blk.body:
            for var in inst.list_vars():
                used_var[var] = inst
    for blk in blocks.values():
        scope = blk.scope
        for var, inst in used_var.items():
            if var.name not in scope.localvars:
                scope.localvars.define(var.name, var)