from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP
def mark_variables(variables, gradients, grad_reqs='write'):
    """Mark NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    variables: list of NDArray
    gradients: list of NDArray
    grad_reqs: list of string
    """
    if isinstance(grad_reqs, string_types):
        grad_reqs = [_GRAD_REQ_MAP[grad_reqs]] * len(variables)
    else:
        grad_reqs = [_GRAD_REQ_MAP[i] for i in grad_reqs]
    check_call(_LIB.MXAutogradMarkVariables(len(variables), c_handle_array(variables), c_array_buf(mx_uint, array('I', grad_reqs)), c_handle_array(gradients)))