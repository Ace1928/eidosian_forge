import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def set_np_shape(active):
    """Turns on/off NumPy shape semantics, in which `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent the shapes
    of zero-size tensors. This is turned off by default for keeping backward compatibility.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Parameters
    ----------
    active : bool
        Indicates whether to turn on/off NumPy shape semantics.

    Returns
    -------
        A bool value indicating the previous state of NumPy shape semantics.

    Example
    -------
    >>> import mxnet as mx
    >>> prev_state = mx.set_np_shape(True)
    >>> print(prev_state)
    False
    >>> print(mx.is_np_shape())
    True
    """
    global _set_np_shape_logged
    if active:
        if not _set_np_shape_logged:
            import logging
            logging.info('NumPy-shape semantics has been activated in your code. This is required for creating and manipulating scalar and zero-size tensors, which were not supported in MXNet before, as in the official NumPy library. Please DO NOT manually deactivate this semantics while using `mxnet.numpy` and `mxnet.numpy_extension` modules.')
            _set_np_shape_logged = True
    elif is_np_array():
        raise ValueError('Deactivating NumPy shape semantics while NumPy array semantics is still active is not allowed. Please consider calling `npx.reset_np()` to deactivate both of them.')
    prev = ctypes.c_int()
    check_call(_LIB.MXSetIsNumpyShape(ctypes.c_int(active), ctypes.byref(prev)))
    return bool(prev.value)