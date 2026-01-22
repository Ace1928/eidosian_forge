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
def replace_var_callback(var, vardict):
    assert isinstance(var, ir.Var)
    while var.name in vardict.keys():
        assert vardict[var.name].name != var.name
        new_var = vardict[var.name]
        var = ir.Var(new_var.scope, new_var.name, new_var.loc)
    return var