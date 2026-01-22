from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/binary_search.h'])
def binary_search(env, exec_policy, first, last, *args):
    """Attempts to find the element value with binary search.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if 1 <= len(args) <= 2:
        value = args[0]
        comp = args[1] if len(args) == 2 else None
        _assert_pointer_of(first, value)
        result_ctype = _cuda_types.bool_
    elif 3 <= len(args) <= 4:
        value_first = args[0]
        value_last = args[1]
        result = args[2]
        comp = args[3] if len(args) == 4 else None
        _assert_same_pointer_type(first, value_first)
        _assert_same_type(value_first, value_last)
        result_ctype = result.ctype
    else:
        raise TypeError('Invalid number of inputs of thrust.binary_search')
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, *args]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::binary_search({params})', result_ctype)