import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def wrap_np_binary_func(func):
    """A convenience decorator for wrapping numpy-compatible binary ufuncs to provide uniform
    error handling.

    Parameters
    ----------
    func : a numpy-compatible binary function to be wrapped for better error handling.

    Returns
    -------
    Function
        A function wrapped with proper error handling.
    """

    @functools.wraps(func)
    def _wrap_np_binary_func(x1, x2, out=None, **kwargs):
        if len(kwargs) != 0:
            for key, value in kwargs.items():
                if key not in _np_ufunc_default_kwargs:
                    raise TypeError("{} is an invalid keyword to function '{}'".format(key, func.__name__))
                if value != _np_ufunc_default_kwargs[key]:
                    if np_ufunc_legal_option(key, value):
                        raise NotImplementedError('{}={} is not implemented yet'.format(key, str(value)))
                    raise TypeError('{} {} not understood'.format(key, value))
        return func(x1, x2, out=out)
    return _wrap_np_binary_func