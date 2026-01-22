import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def handle_border(slice_fn_ty, dim, scope, loc, slice_func_var, stmts, border_inds, border_tuple_items, other_arg, other_first):
    sig = self.typingctx.resolve_function_type(slice_fn_ty, (types.intp,) * 2, {})
    si = border_inds[dim]
    assert isinstance(si, (int, ir.Var))
    si_var = ir.Var(scope, mk_unique_var('$border_ind'), loc)
    self.typemap[si_var.name] = types.intp
    if isinstance(si, int):
        si_assign = ir.Assign(ir.Const(si, loc), si_var, loc)
    else:
        si_assign = ir.Assign(si, si_var, loc)
    stmts.append(si_assign)
    slice_callexpr = ir.Expr.call(func=slice_func_var, args=(other_arg, si_var) if other_first else (si_var, other_arg), kws=(), loc=loc)
    self.calltypes[slice_callexpr] = sig
    border_slice_var = ir.Var(scope, mk_unique_var('$slice'), loc)
    self.typemap[border_slice_var.name] = types.slice2_type
    slice_assign = ir.Assign(slice_callexpr, border_slice_var, loc)
    stmts.append(slice_assign)
    border_tuple_items[dim] = border_slice_var
    border_ind_var = ir.Var(scope, mk_unique_var('$border_index_tuple_var'), loc)
    self.typemap[border_ind_var.name] = types.containers.UniTuple(types.slice2_type, ndims)
    tuple_call = ir.Expr.build_tuple(border_tuple_items, loc)
    tuple_assign = ir.Assign(tuple_call, border_ind_var, loc)
    stmts.append(tuple_assign)
    setitem_call = ir.SetItem(out_arr, border_ind_var, zero_var, loc)
    self.calltypes[setitem_call] = signature(types.none, self.typemap[out_arr.name], self.typemap[border_ind_var.name], self.typemap[out_arr.name].dtype)
    stmts.append(setitem_call)