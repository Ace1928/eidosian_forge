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
def raise_on_unsupported_feature(func_ir, typemap):
    """
    Helper function to walk IR and raise if it finds op codes
    that are unsupported. Could be extended to cover IR sequences
    as well as op codes. Intended use is to call it as a pipeline
    stage just prior to lowering to prevent LoweringErrors for known
    unsupported features.
    """
    gdb_calls = []
    for arg_name in func_ir.arg_names:
        if arg_name in typemap and isinstance(typemap[arg_name], types.containers.UniTuple) and (typemap[arg_name].count > 1000):
            msg = "Tuple '{}' length must be smaller than 1000.\nLarge tuples lead to the generation of a prohibitively large LLVM IR which causes excessive memory pressure and large compile times.\nAs an alternative, the use of a 'list' is recommended in place of a 'tuple' as lists do not suffer from this problem.".format(arg_name)
            raise UnsupportedError(msg, func_ir.loc)
    for blk in func_ir.blocks.values():
        for stmt in blk.find_insts(ir.Assign):
            if isinstance(stmt.value, ir.Expr):
                if stmt.value.op == 'make_function':
                    val = stmt.value
                    code = getattr(val, 'code', None)
                    if code is not None:
                        if getattr(val, 'closure', None) is not None:
                            use = '<creating a function from a closure>'
                            expr = ''
                        else:
                            use = code.co_name
                            expr = '(%s) ' % use
                    else:
                        use = '<could not ascertain use case>'
                        expr = ''
                    msg = 'Numba encountered the use of a language feature it does not support in this context: %s (op code: make_function not supported). If the feature is explicitly supported it is likely that the result of the expression %sis being used in an unsupported manner.' % (use, expr)
                    raise UnsupportedError(msg, stmt.value.loc)
            if isinstance(stmt.value, (ir.Global, ir.FreeVar)):
                val = stmt.value
                val = getattr(val, 'value', None)
                if val is None:
                    continue
                found = False
                if isinstance(val, pytypes.FunctionType):
                    found = val in {numba.gdb, numba.gdb_init}
                if not found:
                    found = getattr(val, '_name', '') == 'gdb_internal'
                if found:
                    gdb_calls.append(stmt.loc)
            if isinstance(stmt.value, ir.Expr):
                if stmt.value.op == 'getattr' and stmt.value.attr == 'view':
                    var = stmt.value.value.name
                    if isinstance(typemap[var], types.Array):
                        continue
                    df = func_ir.get_definition(var)
                    cn = guard(find_callname, func_ir, df)
                    if cn and cn[1] == 'numpy':
                        ty = getattr(numpy, cn[0])
                        if numpy.issubdtype(ty, numpy.integer) or numpy.issubdtype(ty, numpy.floating):
                            continue
                    vardescr = '' if var.startswith('$') else "'{}' ".format(var)
                    raise TypingError("'view' can only be called on NumPy dtypes, try wrapping the variable {}with 'np.<dtype>()'".format(vardescr), loc=stmt.loc)
            if isinstance(stmt.value, ir.Global):
                ty = typemap[stmt.target.name]
                msg = "The use of a %s type, assigned to variable '%s' in globals, is not supported as globals are considered compile-time constants and there is no known way to compile a %s type as a constant."
                if getattr(ty, 'reflected', False) or isinstance(ty, (types.DictType, types.ListType)):
                    raise TypingError(msg % (ty, stmt.value.name, ty), loc=stmt.loc)
            if isinstance(stmt.value, ir.Yield) and (not func_ir.is_generator):
                msg = 'The use of generator expressions is unsupported.'
                raise UnsupportedError(msg, loc=stmt.loc)
    if len(gdb_calls) > 1:
        msg = 'Calling either numba.gdb() or numba.gdb_init() more than once in a function is unsupported (strange things happen!), use numba.gdb_breakpoint() to create additional breakpoints instead.\n\nRelevant documentation is available here:\nhttps://numba.readthedocs.io/en/stable/user/troubleshoot.html#using-numba-s-direct-gdb-bindings-in-nopython-mode\n\nConflicting calls found at:\n %s'
        buf = '\n'.join([x.strformat() for x in gdb_calls])
        raise UnsupportedError(msg % buf)