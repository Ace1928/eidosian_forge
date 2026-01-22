import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def do_prune(take_truebr, blk):
    keep = branch.truebr if take_truebr else branch.falsebr
    jmp = ir.Jump(keep, loc=branch.loc)
    blk.body[-1] = jmp
    return 1 if keep == branch.truebr else 0