from numba import njit
from numba.tests.gdb_support import GdbMIDriver
from numba.tests.support import TestCase, needs_subprocess
import unittest
def foo_factory(n):

    @njit(debug=True)
    def foo(x):
        z = 7 + n
        return (x, z)
    return foo