import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def find_literally_calls(func_ir, argtypes):
    """An analysis to find `numba.literally` call inside the given IR.
    When an unsatisfied literal typing request is found, a `ForceLiteralArg`
    exception is raised.

    Parameters
    ----------

    func_ir : numba.ir.FunctionIR

    argtypes : Sequence[numba.types.Type]
        The argument types.
    """
    from numba.core import ir_utils
    marked_args = set()
    first_loc = {}
    for blk in func_ir.blocks.values():
        for assign in blk.find_exprs(op='call'):
            var = ir_utils.guard(ir_utils.get_definition, func_ir, assign.func)
            if isinstance(var, (ir.Global, ir.FreeVar)):
                fnobj = var.value
            else:
                fnobj = ir_utils.guard(ir_utils.resolve_func_from_module, func_ir, var)
            if fnobj is special.literally:
                [arg] = assign.args
                defarg = func_ir.get_definition(arg)
                if isinstance(defarg, ir.Arg):
                    argindex = defarg.index
                    marked_args.add(argindex)
                    first_loc.setdefault(argindex, assign.loc)
    for pos in marked_args:
        query_arg = argtypes[pos]
        do_raise = isinstance(query_arg, types.InitialValue) and query_arg.initial_value is None
        if do_raise:
            loc = first_loc[pos]
            raise errors.ForceLiteralArg(marked_args, loc=loc)
        if not isinstance(query_arg, (types.Literal, types.InitialValue)):
            loc = first_loc[pos]
            raise errors.ForceLiteralArg(marked_args, loc=loc)