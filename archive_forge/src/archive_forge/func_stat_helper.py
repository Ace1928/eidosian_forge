import re
import ctypes
import logging
from math import sqrt
from .ndarray import NDArray
from .base import NDArrayHandle, py_str
from . import ndarray
def stat_helper(name, array):
    """wrapper for executor callback"""
    array = ctypes.cast(array, NDArrayHandle)
    array = NDArray(array, writable=False)
    if not self.activated or not self.re_prog.match(py_str(name)):
        return
    self.queue.append((self.step, py_str(name), self.stat_func(array)))