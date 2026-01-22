import unittest
from numba.tests.support import TestCase
from numba.misc.init_utils import version_info, generate_version_info
def test_full_rev(self):
    expected = version_info(0, 1, None, (0, 1), (0, 1, None), '0.1.0dev0+1.g0123456789abcdef', ('0', '1', '0dev0+1', 'g0123456789abcdef'), 'g0123456789abcdef')
    received = generate_version_info('0.1.0dev0+1.g0123456789abcdef')
    self.assertEqual(received, expected)