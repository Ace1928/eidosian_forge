import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_STORE_SLICE_0(self, state, inst):
    """
        TOS[:] = TOS1
        """
    tos = state.pop()
    value = state.pop()
    slicevar = state.make_temp()
    indexvar = state.make_temp()
    nonevar = state.make_temp()
    state.append(inst, base=tos, value=value, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)