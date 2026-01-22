import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def prune_by_predicate(branch, pred, blk):
    try:
        if not isinstance(pred, (ir.Const, ir.FreeVar, ir.Global)):
            raise TypeError('Expected constant Numba IR node')
        take_truebr = bool(pred.value)
    except TypeError:
        return (False, None)
    if DEBUG > 0:
        kill = branch.falsebr if take_truebr else branch.truebr
        print('Pruning %s' % kill, branch, pred)
    taken = do_prune(take_truebr, blk)
    return (True, taken)