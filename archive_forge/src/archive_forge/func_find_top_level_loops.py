import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def find_top_level_loops(cfg):
    """
    A generator that yields toplevel loops given a control-flow-graph
    """
    blocks_in_loop = set()
    for loop in cfg.loops().values():
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        insiders.discard(loop.header)
        blocks_in_loop |= insiders
    for loop in cfg.loops().values():
        if loop.header not in blocks_in_loop:
            yield _fix_loop_exit(cfg, loop)