from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/scan.h'])
def inclusive_scan(env, exec_policy, first, last, result, binary_op=None):
    """Computes an inclusive prefix sum operation.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::inclusive_scan({params})', result.ctype)