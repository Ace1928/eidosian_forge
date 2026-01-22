from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def replace_target(term, src, dst):

    def replace(target):
        return dst if target == src else target
    if isinstance(term, ir.Branch):
        return ir.Branch(cond=term.cond, truebr=replace(term.truebr), falsebr=replace(term.falsebr), loc=term.loc)
    elif isinstance(term, ir.Jump):
        return ir.Jump(target=replace(term.target), loc=term.loc)
    else:
        assert not term.get_targets()
        return term