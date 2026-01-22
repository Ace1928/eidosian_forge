from numba import njit
from numba.tests.gdb_support import GdbMIDriver
from numba.tests.support import TestCase, needs_subprocess
import unittest
@njit(debug=True)
def call_foo():
    a = foo1(10)
    b = foo2(20)
    c = foo3(30)
    return (a, b, c)