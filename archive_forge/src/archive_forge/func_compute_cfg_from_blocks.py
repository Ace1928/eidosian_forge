import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def compute_cfg_from_blocks(blocks):
    cfg = CFGraph()
    for k in blocks:
        cfg.add_node(k)
    for k, b in blocks.items():
        term = b.terminator
        for target in term.get_targets():
            cfg.add_edge(k, target)
    cfg.set_entry_point(min(blocks))
    cfg.process()
    return cfg