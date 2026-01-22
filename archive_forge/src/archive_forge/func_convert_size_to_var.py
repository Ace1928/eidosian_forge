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
def convert_size_to_var(size_var, typemap, scope, loc, nodes):
    if isinstance(size_var, int):
        new_size = ir.Var(scope, mk_unique_var('$alloc_size'), loc)
        if typemap:
            typemap[new_size.name] = types.intp
        size_assign = ir.Assign(ir.Const(size_var, loc), new_size, loc)
        nodes.append(size_assign)
        return new_size
    assert isinstance(size_var, ir.Var)
    return size_var