import unittest
from cvxpy.utilities.versioning import Version
def test_tuple_construction(self):
    self.assertTrue(Version('0100.2.03') == Version((100, 2, 3)))
    self.assertTrue(Version('1.2.3') == Version((1, 2, 3, None)))
    self.assertTrue(Version('1.2.3') == Version((1, 2, 3, 'junk')))
    self.assertTrue(Version('1.2.3') == Version((1, 2, 3, -1)))