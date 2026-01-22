import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_LIST_TO_TUPLE(self, state, inst):
    tos = state.pop()
    res = state.make_temp()
    state.append(inst, const_list=tos, res=res)
    state.push(res)