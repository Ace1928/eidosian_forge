from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
def test_unicode_name2(self):
    fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
    cfn = njit(fn)
    self.assertEqual(cfn(1, 2), 3)