from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/binary_search.h'])
def equal_range(env, exec_policy, first, last, value, comp=None):
    """Attempts to find the element value in an ordered range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::equal_range({params})', _cuda_types.Tuple([first.ctype, first.ctype]))