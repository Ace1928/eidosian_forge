import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def gen_wrap_if_not_known(val, val_typ, known):
    if not known:
        var = ir.Var(scope, mk_unique_var('var'), loc)
        var_typ = types.intp
        new_value = ir.Expr.call(wrap_var, [val, dsize], {}, loc)
        self._define(equiv_set, var, var_typ, new_value)
        self.calltypes[new_value] = sig
        return (var, var_typ, new_value)
    else:
        return (val, val_typ, None)