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
def get_np_ufunc_typ(func):
    """get type of the incoming function from builtin registry"""
    for k, v in typing.npydecl.registry.globals:
        if k == func:
            return v
    for k, v in typing.templates.builtin_registry.globals:
        if k == func:
            return v
    raise RuntimeError('type for func ', func, ' not found')