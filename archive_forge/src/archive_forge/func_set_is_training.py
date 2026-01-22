from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP
def set_is_training(is_train):
    """Set status to training/not training. When training, graph will be constructed
    for gradient computation. Operators will also run with ctx.is_train=True. For example,
    Dropout will drop inputs randomly when is_train=True while simply passing through
    if is_train=False.

    Parameters
    ----------
    is_train: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(ctypes.c_int(is_train), ctypes.byref(prev)))
    check_call(_LIB.MXAutogradSetIsRecording(ctypes.c_int(is_train), ctypes.byref(prev)))
    return bool(prev.value)