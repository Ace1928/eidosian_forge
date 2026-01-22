import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_DUP_TOPX(self, state, inst):
    count = inst.arg
    assert 1 <= count <= 5, 'Invalid DUP_TOPX count'
    self._dup_topx(state, inst, count)