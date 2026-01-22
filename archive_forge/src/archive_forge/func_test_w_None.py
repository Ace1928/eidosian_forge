import unittest
from zope.interface import Interface
def test_w_None(self):
    l = [I1, None, I3, I5, I6, I4, I2]
    l.sort()
    self.assertEqual(l, [I1, I2, I3, I4, I5, I6, None])