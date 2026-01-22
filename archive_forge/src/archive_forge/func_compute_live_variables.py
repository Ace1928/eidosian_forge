import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def compute_live_variables(cfg, blocks, var_def_map, var_dead_map):
    """
    Compute the live variables at the beginning of each block
    and at each yield point.
    The ``var_def_map`` and ``var_dead_map`` indicates the variable defined
    and deleted at each block, respectively.
    """
    block_entry_vars = defaultdict(set)

    def fix_point_progress():
        return tuple(map(len, block_entry_vars.values()))
    old_point = None
    new_point = fix_point_progress()
    while old_point != new_point:
        for offset in blocks:
            avail = block_entry_vars[offset] | var_def_map[offset]
            avail -= var_dead_map[offset]
            for succ, _data in cfg.successors(offset):
                block_entry_vars[succ] |= avail
        old_point = new_point
        new_point = fix_point_progress()
    return block_entry_vars