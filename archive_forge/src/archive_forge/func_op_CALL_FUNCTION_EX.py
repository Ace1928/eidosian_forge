import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_CALL_FUNCTION_EX(self, state, inst):
    if inst.arg & 1 and PYVERSION < (3, 10):
        errmsg = 'CALL_FUNCTION_EX with **kwargs not supported'
        raise UnsupportedError(errmsg)
    if inst.arg & 1:
        varkwarg = state.pop()
    else:
        varkwarg = None
    vararg = state.pop()
    func = state.pop()
    if PYVERSION in ((3, 11), (3, 12)):
        if _is_null_temp_reg(state.peek(1)):
            state.pop()
    elif PYVERSION in ((3, 9), (3, 10)):
        pass
    else:
        raise NotImplementedError(PYVERSION)
    res = state.make_temp()
    state.append(inst, func=func, vararg=vararg, varkwarg=varkwarg, res=res)
    state.push(res)