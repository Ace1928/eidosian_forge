import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
@guvectorize([(float64[:], float64[:])], '()->()', writable_args=('invals',))
def init_values(invals, outvals):
    invals[0] = 6.5
    outvals[0] = 4.2