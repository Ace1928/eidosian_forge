import time
import ctypes
import numpy as np
from numba.tests.support import captured_stdout
from numba import vectorize, guvectorize
import unittest
@vectorize('float64(float64, float64)', target='parallel')
def fnv(a, b):
    return a + b