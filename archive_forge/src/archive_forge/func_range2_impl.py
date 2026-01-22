import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
@lower_builtin(range, int_type, int_type)
@lower_builtin(prange, int_type, int_type)
@lower_builtin(internal_prange, int_type, int_type)
def range2_impl(context, builder, sig, args):
    """
        range(start: int, stop: int) -> range object
        """
    start, stop = args
    state = RangeState(context, builder)
    state.start = start
    state.stop = stop
    state.step = context.get_constant(int_type, 1)
    return impl_ret_untracked(context, builder, range_state_type, state._getvalue())