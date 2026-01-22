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
def legalize_names(varnames):
    """returns a dictionary for conversion of variable names to legal
    parameter names.
    """
    var_map = {}
    for var in varnames:
        new_name = var.replace('_', '__').replace('$', '_').replace('.', '_')
        assert new_name not in var_map
        var_map[var] = new_name
    return var_map