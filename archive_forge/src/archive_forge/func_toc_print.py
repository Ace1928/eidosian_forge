import re
import ctypes
import logging
from math import sqrt
from .ndarray import NDArray
from .base import NDArrayHandle, py_str
from . import ndarray
def toc_print(self):
    """End collecting and print results."""
    res = self.toc()
    for n, k, v in res:
        logging.info('Batch: {:7d} {:30s} {:s}'.format(n, k, v))