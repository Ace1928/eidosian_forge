from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP
def train_section():
    """Returns a training scope context to be used in 'with' statement
    and captures training code.

    Example::
        with autograd.train_section():
            y = model(x)
            compute_gradient([y])
        metric.update(...)
        optim.step(...)
    """
    return TrainingStateScope(True)