from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
def test_cfunc(self):
    fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
    cfn = cfunc('int32(int32, int32)')(fn)
    self.assertEqual(cfn.ctypes(1, 2), 3)