import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_MAKE_FUNCTION(self, state, inst, MAKE_CLOSURE=False):
    if PYVERSION in ((3, 11), (3, 12)):
        name = None
    elif PYVERSION in ((3, 9), (3, 10)):
        name = state.pop()
    else:
        raise NotImplementedError(PYVERSION)
    code = state.pop()
    closure = annotations = kwdefaults = defaults = None
    if inst.arg & 8:
        closure = state.pop()
    if inst.arg & 4:
        annotations = state.pop()
    if inst.arg & 2:
        kwdefaults = state.pop()
    if inst.arg & 1:
        defaults = state.pop()
    res = state.make_temp()
    state.append(inst, name=name, code=code, closure=closure, annotations=annotations, kwdefaults=kwdefaults, defaults=defaults, res=res)
    state.push(res)