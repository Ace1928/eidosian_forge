import os
import sys
import ctypes
from ..base import _LIB, check_call
from .base import py_str, c_str
def get_global_func(name, allow_missing=False):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : tvm.Function
        The function to be returned, None if function is missing.
    """
    handle = FunctionHandle()
    check_call(_LIB.MXNetFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return Function(handle, False)
    if allow_missing:
        return None
    raise ValueError('Cannot find global function %s' % name)