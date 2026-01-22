import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BUILD_CONST_KEY_MAP(self, state, inst):
    keys = state.pop()
    vals = list(reversed([state.pop() for _ in range(inst.arg)]))
    keytmps = [state.make_temp() for _ in range(inst.arg)]
    res = state.make_temp()
    state.append(inst, keys=keys, keytmps=keytmps, values=vals, res=res)
    state.push(res)