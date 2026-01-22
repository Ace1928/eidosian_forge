from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def same_exit_point(loop):
    """all exits must point to the same location"""
    outedges = set()
    for k in loop.exits:
        succs = set((x for x, _ in cfg.successors(k)))
        if not succs:
            _logger.debug('return-statement in loop.')
            return False
        outedges |= succs
    ok = len(outedges) == 1
    _logger.debug('same_exit_point=%s (%s)', ok, outedges)
    return ok