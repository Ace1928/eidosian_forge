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
def find_potential_aliases(blocks, args, typemap, func_ir, alias_map=None, arg_aliases=None):
    """find all array aliases and argument aliases to avoid remove as dead"""
    if alias_map is None:
        alias_map = {}
    if arg_aliases is None:
        arg_aliases = set((a for a in args if not is_immutable_type(a, typemap)))
    func_ir._definitions = build_definitions(func_ir.blocks)
    np_alias_funcs = ['ravel', 'transpose', 'reshape']
    for bl in blocks.values():
        for instr in bl.body:
            if type(instr) in alias_analysis_extensions:
                f = alias_analysis_extensions[type(instr)]
                f(instr, args, typemap, func_ir, alias_map, arg_aliases)
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target.name
                if is_immutable_type(lhs, typemap):
                    continue
                if isinstance(expr, ir.Var) and lhs != expr.name:
                    _add_alias(lhs, expr.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and (expr.op == 'cast' or expr.op in ['getitem', 'static_getitem']):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and expr.op == 'inplace_binop':
                    _add_alias(lhs, expr.lhs.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and expr.op == 'getattr' and (expr.attr in ['T', 'ctypes', 'flat']):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and expr.op == 'getattr' and (expr.attr not in ['shape']) and (expr.value.name in arg_aliases):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and expr.op == 'call':
                    fdef = guard(find_callname, func_ir, expr, typemap)
                    if fdef is None:
                        continue
                    fname, fmod = fdef
                    if fdef in alias_func_extensions:
                        alias_func = alias_func_extensions[fdef]
                        alias_func(lhs, expr.args, alias_map, arg_aliases)
                    if fmod == 'numpy' and fname in np_alias_funcs:
                        _add_alias(lhs, expr.args[0].name, alias_map, arg_aliases)
                    if isinstance(fmod, ir.Var) and fname in np_alias_funcs:
                        _add_alias(lhs, fmod.name, alias_map, arg_aliases)
    old_alias_map = copy.deepcopy(alias_map)
    for v in old_alias_map:
        for w in old_alias_map[v]:
            alias_map[v] |= alias_map[w]
        for w in old_alias_map[v]:
            alias_map[w] = alias_map[v]
    return (alias_map, arg_aliases)