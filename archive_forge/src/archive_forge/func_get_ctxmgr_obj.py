from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def get_ctxmgr_obj(var_ref):
    """Return the context-manager object and extra info.

        The extra contains the arguments if the context-manager is used
        as a call.
        """
    dfn = func_ir.get_definition(var_ref)
    if isinstance(dfn, ir.Expr) and dfn.op == 'call':
        args = [get_var_dfn(x) for x in dfn.args]
        kws = {k: get_var_dfn(v) for k, v in dfn.kws}
        extra = {'args': args, 'kwargs': kws}
        var_ref = dfn.func
    else:
        extra = None
    ctxobj = ir_utils.guard(ir_utils.find_outer_value, func_ir, var_ref)
    if ctxobj is ir.UNDEFINED:
        raise errors.CompilerError('Undefined variable used as context manager', loc=blocks[blk_start].loc)
    if ctxobj is None:
        raise errors.CompilerError(_illegal_cm_msg, loc=dfn.loc)
    return (ctxobj, extra)