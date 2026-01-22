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
def canonicalize_array_math(func_ir, typemap, calltypes, typingctx):
    blocks = func_ir.blocks
    saved_arr_arg = {}
    topo_order = find_topo_order(blocks)
    for label in topo_order:
        block = blocks[label]
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                rhs = stmt.value
                if rhs.op == 'getattr' and rhs.attr in arr_math and isinstance(typemap[rhs.value.name], types.npytypes.Array):
                    rhs = stmt.value
                    arr = rhs.value
                    saved_arr_arg[lhs] = arr
                    scope = arr.scope
                    loc = arr.loc
                    g_np_var = ir.Var(scope, mk_unique_var('$np_g_var'), loc)
                    typemap[g_np_var.name] = types.misc.Module(numpy)
                    g_np = ir.Global('np', numpy, loc)
                    g_np_assign = ir.Assign(g_np, g_np_var, loc)
                    rhs.value = g_np_var
                    new_body.append(g_np_assign)
                    func_ir._definitions[g_np_var.name] = [g_np]
                    func = getattr(numpy, rhs.attr)
                    func_typ = get_np_ufunc_typ(func)
                    typemap.pop(lhs)
                    typemap[lhs] = func_typ
                if rhs.op == 'call' and rhs.func.name in saved_arr_arg:
                    arr = saved_arr_arg[rhs.func.name]
                    old_sig = calltypes.pop(rhs)
                    argtyps = old_sig.args[:len(rhs.args)]
                    kwtyps = {name: typemap[v.name] for name, v in rhs.kws}
                    calltypes[rhs] = typemap[rhs.func.name].get_call_type(typingctx, [typemap[arr.name]] + list(argtyps), kwtyps)
                    rhs.args = [arr] + rhs.args
            new_body.append(stmt)
        block.body = new_body
    return