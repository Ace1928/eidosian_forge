from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def find_region_inout_vars(blocks, livemap, callfrom, returnto, body_block_ids):
    """Find input and output variables to a block region.
    """
    inputs = livemap[callfrom]
    outputs = livemap[returnto]
    loopblocks = {}
    for k in body_block_ids:
        loopblocks[k] = blocks[k]
    used_vars = set()
    def_vars = set()
    defs = compute_use_defs(loopblocks)
    for vs in defs.usemap.values():
        used_vars |= vs
    for vs in defs.defmap.values():
        def_vars |= vs
    used_or_defined = used_vars | def_vars
    inputs = sorted(set(inputs) & used_or_defined)
    outputs = sorted(set(outputs) & used_or_defined & def_vars)
    return (inputs, outputs)