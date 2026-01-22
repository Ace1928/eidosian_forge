import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def peep_hole_call_function_ex_to_call_function_kw(func_ir):
    """
    This peephole rewrites a bytecode sequence unique to Python 3.10
    where CALL_FUNCTION_EX is used instead of CALL_FUNCTION_KW because of
    stack limitations set by CPython. This limitation is imposed whenever
    a function call has too many arguments or keyword arguments.

    https://github.com/python/cpython/blob/a58ebcc701dd6c43630df941481475ff0f615a81/Python/compile.c#L55
    https://github.com/python/cpython/blob/a58ebcc701dd6c43630df941481475ff0f615a81/Python/compile.c#L4442

    In particular, this change is imposed whenever (n_args / 2) + n_kws > 15.

    Different bytecode is generated for args depending on if n_args > 30
    or n_args <= 30 and similarly if n_kws > 15 or n_kws <= 15.

    This function unwraps the *args and **kwargs in the function call
    and places these values directly into the args and kwargs of the call.
    """
    errmsg = textwrap.dedent('\n        CALL_FUNCTION_EX with **kwargs not supported.\n        If you are not using **kwargs this may indicate that\n        you have a large number of kwargs and are using inlined control\n        flow. You can resolve this issue by moving the control flow out of\n        the function call. For example, if you have\n\n            f(a=1 if flag else 0, ...)\n\n        Replace that with:\n\n            a_val = 1 if flag else 0\n            f(a=a_val, ...)')
    already_deleted_defs = collections.defaultdict(set)
    for blk in func_ir.blocks.values():
        blk_changed = False
        new_body = []
        for i, stmt in enumerate(blk.body):
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'call') and (stmt.value.varkwarg is not None):
                blk_changed = True
                call = stmt.value
                args = call.args
                kws = call.kws
                vararg = call.vararg
                varkwarg = call.varkwarg
                start_search = i - 1
                varkwarg_loc = start_search
                keyword_def = None
                found = False
                while varkwarg_loc >= 0 and (not found):
                    keyword_def = blk.body[varkwarg_loc]
                    if isinstance(keyword_def, ir.Assign) and keyword_def.target.name == varkwarg.name:
                        found = True
                    else:
                        varkwarg_loc -= 1
                if kws or not found or (not (isinstance(keyword_def.value, ir.Expr) and keyword_def.value.op == 'build_map')):
                    raise UnsupportedError(errmsg)
                if keyword_def.value.items:
                    kws = _call_function_ex_replace_kws_small(blk.body, keyword_def.value, new_body, varkwarg_loc, func_ir, already_deleted_defs)
                else:
                    kws = _call_function_ex_replace_kws_large(blk.body, varkwarg.name, varkwarg_loc, i - 1, new_body, func_ir, errmsg, already_deleted_defs)
                start_search = varkwarg_loc
                if vararg is not None:
                    if args:
                        raise UnsupportedError(errmsg)
                    vararg_loc = start_search
                    args_def = None
                    found = False
                    while vararg_loc >= 0 and (not found):
                        args_def = blk.body[vararg_loc]
                        if isinstance(args_def, ir.Assign) and args_def.target.name == vararg.name:
                            found = True
                        else:
                            vararg_loc -= 1
                    if not found:
                        raise UnsupportedError(errmsg)
                    if isinstance(args_def.value, ir.Expr) and args_def.value.op == 'build_tuple':
                        args = _call_function_ex_replace_args_small(blk.body, args_def.value, new_body, vararg_loc, func_ir, already_deleted_defs)
                    elif isinstance(args_def.value, ir.Expr) and args_def.value.op == 'list_to_tuple':
                        raise UnsupportedError(errmsg)
                    else:
                        args = _call_function_ex_replace_args_large(blk.body, args_def, new_body, vararg_loc, func_ir, errmsg, already_deleted_defs)
                new_call = ir.Expr.call(call.func, args, kws, call.loc, target=call.target)
                _remove_assignment_definition(blk.body, i, func_ir, already_deleted_defs)
                stmt = ir.Assign(new_call, stmt.target, stmt.loc)
                func_ir._definitions[stmt.target.name].append(new_call)
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'call') and (stmt.value.vararg is not None):
                call = stmt.value
                vararg_name = call.vararg.name
                if vararg_name in func_ir._definitions and len(func_ir._definitions[vararg_name]) == 1:
                    expr = func_ir._definitions[vararg_name][0]
                    if isinstance(expr, ir.Expr) and expr.op == 'list_to_tuple':
                        raise UnsupportedError(errmsg)
            new_body.append(stmt)
        if blk_changed:
            blk.body.clear()
            blk.body.extend([x for x in new_body if x is not None])
    return func_ir