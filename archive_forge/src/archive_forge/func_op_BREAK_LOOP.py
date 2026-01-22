import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BREAK_LOOP(self, state, inst):
    end = state.get_top_block('LOOP')['end']
    state.append(inst, end=end)
    state.pop_block()
    state.fork(pc=end)