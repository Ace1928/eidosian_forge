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
def set_index_var_of_get_setitem(stmt, new_index):
    if is_getitem(stmt):
        if stmt.value.op == 'getitem':
            stmt.value.index = new_index
        else:
            stmt.value.index_var = new_index
    elif is_setitem(stmt):
        if isinstance(stmt, ir.SetItem):
            stmt.index = new_index
        else:
            stmt.index_var = new_index
    else:
        raise ValueError('getitem or setitem node expected but received {}'.format(stmt))