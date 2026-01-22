import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def rewrite_semantic_constants(func_ir, called_args):
    """
    This rewrites values known to be constant by their semantics as ir.Const
    nodes, this is to give branch pruning the best chance possible of killing
    branches. An example might be rewriting len(tuple) as the literal length.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
    DEBUG = 0
    if DEBUG > 1:
        print(('rewrite_semantic_constants: ' + func_ir.func_id.func_name).center(80, '-'))
        print('before'.center(80, '*'))
        func_ir.dump()

    def rewrite_statement(func_ir, stmt, new_val):
        """
        Rewrites the stmt as a ir.Const new_val and fixes up the entries in
        func_ir._definitions
        """
        stmt.value = ir.Const(new_val, stmt.loc)
        defns = func_ir._definitions[stmt.target.name]
        repl_idx = defns.index(val)
        defns[repl_idx] = stmt.value

    def rewrite_array_ndim(val, func_ir, called_args):
        if getattr(val, 'op', None) == 'getattr':
            if val.attr == 'ndim':
                arg_def = guard(get_definition, func_ir, val.value)
                if isinstance(arg_def, ir.Arg):
                    argty = called_args[arg_def.index]
                    if isinstance(argty, types.Array):
                        rewrite_statement(func_ir, stmt, argty.ndim)

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
    from numba.core.ir_utils import get_definition, guard
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                val = stmt.value
                if isinstance(val, ir.Expr):
                    rewrite_array_ndim(val, func_ir, called_args)
                    rewrite_tuple_len(val, func_ir, called_args)
    if DEBUG > 1:
        print('after'.center(80, '*'))
        func_ir.dump()
        print('-' * 80)