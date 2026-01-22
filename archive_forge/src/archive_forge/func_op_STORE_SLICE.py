import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_STORE_SLICE(self, state, inst):
    end = state.pop()
    start = state.pop()
    container = state.pop()
    value = state.pop()
    slicevar = state.make_temp()
    res = state.make_temp()
    state.append(inst, start=start, end=end, container=container, value=value, res=res, slicevar=slicevar)