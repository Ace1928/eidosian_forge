import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def complex_np_types():
    for tp_name in ('complex64', 'complex128'):
        yield tp_name