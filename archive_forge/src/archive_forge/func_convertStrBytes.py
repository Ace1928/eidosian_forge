from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def convertStrBytes(func):
    """
    In python 3, strings are unicode instead of bytes, and need to be converted for ctypes
    Args from caller: (1, 'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF>)
    Args passed to function: (1, b'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF)>
    ----
    Returned from function: b'returned string'
    Returned to caller: 'returned string'
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [arg.encode() if isinstance(arg, str) else arg for arg in args]
        res = func(*args, **kwargs)
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res
    if sys.version_info >= (3,):
        return wrapper
    return func