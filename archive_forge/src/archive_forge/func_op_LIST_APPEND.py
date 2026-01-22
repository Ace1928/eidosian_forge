import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_LIST_APPEND(self, state, inst):
    value = state.pop()
    index = inst.arg
    target = state.peek(index)
    appendvar = state.make_temp()
    res = state.make_temp()
    state.append(inst, target=target, value=value, appendvar=appendvar, res=res)