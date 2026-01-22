from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def consolidate_multi_exit_withs(withs: dict, blocks, func_ir):
    """Modify the FunctionIR to merge the exit blocks of with constructs.
    """
    for k in withs:
        vs: set = withs[k]
        if len(vs) > 1:
            func_ir, common = _fix_multi_exit_blocks(func_ir, vs, split_condition=ir_utils.is_pop_block)
            withs[k] = {common}
    return func_ir