import unittest
from zope.interface import Interface
def test_I1_I2(self):
    self.assertLess(I1.__name__, I2.__name__)
    self.assertEqual(I1.__module__, I2.__module__)
    self.assertEqual(I1.__module__, __name__)
    self.assertLess(I1, I2)