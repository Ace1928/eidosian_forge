import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def is_np_array():
    """Checks whether the NumPy-array semantics is currently turned on.
    This is currently used in Gluon for checking whether an array of type `mxnet.numpy.ndarray`
    or `mx.nd.NDArray` should be created. For example, at the time when a parameter
    is created in a `Block`, an `mxnet.numpy.ndarray` is created if this returns true; else
    an `mx.nd.NDArray` is created.

    Normally, users are not recommended to use this API directly unless you known exactly
    what is going on under the hood.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Returns
    -------
        A bool value indicating whether the NumPy-array semantics is currently on.
    """
    return _NumpyArrayScope._current.value._is_np_array if hasattr(_NumpyArrayScope._current, 'value') else False