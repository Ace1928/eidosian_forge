from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/swap.h'])
def swap_ranges(env, exec_policy, first1, last1, first2):
    """Swaps each of the elements in the range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first1)
    _assert_same_type(first1, last1)
    _assert_pointer_type(first2)
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::swap_ranges({params})', first2.ctype)