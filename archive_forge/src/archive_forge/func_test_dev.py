import unittest
from numba.tests.support import TestCase
from numba.misc.init_utils import version_info, generate_version_info
def test_dev(self):
    expected = version_info(0, 1, None, (0, 1), (0, 1, None), '0.1.0dev0', ('0', '1', '0dev0'), None)
    received = generate_version_info('0.1.0dev0')
    self.assertEqual(received, expected)