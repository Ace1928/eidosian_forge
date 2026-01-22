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
def get_unused_var_name(prefix, var_table):
    """ Get a new var name with a given prefix and
        make sure it is unused in the given variable table.
    """
    cur = 0
    while True:
        var = prefix + str(cur)
        if var not in var_table:
            return var
        cur += 1