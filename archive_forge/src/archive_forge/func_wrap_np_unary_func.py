import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def wrap_np_unary_func(func):
    """A convenience decorator for wrapping numpy-compatible unary ufuncs to provide uniform
    error handling.

    Parameters
    ----------
    func : a numpy-compatible unary function to be wrapped for better error handling.

    Returns
    -------
    Function
        A function wrapped with proper error handling.
    """

    @functools.wraps(func)
    def _wrap_np_unary_func(x, out=None, **kwargs):
        if len(kwargs) != 0:
            for key, value in kwargs.items():
                if key not in _np_ufunc_default_kwargs:
                    raise TypeError("{} is an invalid keyword to function '{}'".format(key, func.__name__))
                if value != _np_ufunc_default_kwargs[key]:
                    if np_ufunc_legal_option(key, value):
                        raise NotImplementedError('{}={} is not implemented yet for operator {}'.format(key, str(value), func.__name__))
                    raise TypeError('{}={} not understood for operator {}'.format(key, value, func.__name__))
        return func(x, out=out)
    return _wrap_np_unary_func