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
def mk_loop_header(typemap, phi_var, calltypes, scope, loc):
    """make a block that is a loop header updating iteration variables.
    target labels in branch need to be set.
    """
    iternext_var = ir.Var(scope, mk_unique_var('$iternext_var'), loc)
    typemap[iternext_var.name] = types.containers.Pair(types.intp, types.boolean)
    iternext_call = ir.Expr.iternext(phi_var, loc)
    calltypes[iternext_call] = signature(types.containers.Pair(types.intp, types.boolean), types.range_iter64_type)
    iternext_assign = ir.Assign(iternext_call, iternext_var, loc)
    pair_first_var = ir.Var(scope, mk_unique_var('$pair_first_var'), loc)
    typemap[pair_first_var.name] = types.intp
    pair_first_call = ir.Expr.pair_first(iternext_var, loc)
    pair_first_assign = ir.Assign(pair_first_call, pair_first_var, loc)
    pair_second_var = ir.Var(scope, mk_unique_var('$pair_second_var'), loc)
    typemap[pair_second_var.name] = types.boolean
    pair_second_call = ir.Expr.pair_second(iternext_var, loc)
    pair_second_assign = ir.Assign(pair_second_call, pair_second_var, loc)
    phi_b_var = ir.Var(scope, mk_unique_var('$phi'), loc)
    typemap[phi_b_var.name] = types.intp
    phi_b_assign = ir.Assign(pair_first_var, phi_b_var, loc)
    branch = ir.Branch(pair_second_var, -1, -1, loc)
    header_block = ir.Block(scope, loc)
    header_block.body = [iternext_assign, pair_first_assign, pair_second_assign, phi_b_assign, branch]
    return header_block