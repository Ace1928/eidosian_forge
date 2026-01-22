import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def rewrite_tuple_len(val, func_ir, called_args):
    if getattr(val, 'op', None) == 'call':
        func = guard(get_definition, func_ir, val.func)
        if func is not None and isinstance(func, ir.Global) and (getattr(func, 'value', None) is len):
            arg, = val.args
            arg_def = guard(get_definition, func_ir, arg)
            if isinstance(arg_def, ir.Arg):
                argty = called_args[arg_def.index]
                if isinstance(argty, types.BaseTuple):
                    rewrite_statement(func_ir, stmt, argty.count)
            elif isinstance(arg_def, ir.Expr) and arg_def.op == 'typed_getitem':
                argty = arg_def.dtype
                if isinstance(argty, types.BaseTuple):
                    rewrite_statement(func_ir, stmt, argty.count)