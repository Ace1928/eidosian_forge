import unittest
from cvxpy.utilities.versioning import Version
def test_typical_inputs(self):
    self.assertTrue(Version('1.0.0') < Version('2.0.0'))
    self.assertTrue(Version('1.0.0') < Version('1.1.0'))
    self.assertTrue(Version('1.0.0') < Version('1.0.1'))
    self.assertFalse(Version('1.0.0') < Version('1.0.0'))
    self.assertTrue(Version('1.0.0') <= Version('1.0.0'))
    self.assertFalse(Version('1.0.0') >= Version('2.0.0'))
    self.assertFalse(Version('1.0.0') >= Version('1.1.0'))
    self.assertFalse(Version('1.0.0') >= Version('1.0.1'))
    self.assertTrue(Version('1.0.0') >= Version('1.0.0'))
    self.assertFalse(Version('1.0.0') > Version('1.0.0'))