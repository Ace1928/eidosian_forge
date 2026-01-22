from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/replace.h'])
def replace_copy(env, exec_policy, first, last, result, old_value, new_value):
    """Replaces every element in the range equal to old_value with new_value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    _assert_same_type(old_value, new_value)
    args = [exec_policy, first, last, result, old_value, new_value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::replace_copy({params})', result.ctype)