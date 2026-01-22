import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_UNPACK_SEQUENCE(self, state, inst):
    count = inst.arg
    iterable = state.pop()
    stores = [state.make_temp() for _ in range(count)]
    tupleobj = state.make_temp()
    state.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
    for st in reversed(stores):
        state.push(st)