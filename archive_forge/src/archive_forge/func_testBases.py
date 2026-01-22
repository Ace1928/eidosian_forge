import unittest2 as unittest
from mock import sentinel, DEFAULT
def testBases(self):
    self.assertRaises(AttributeError, lambda: sentinel.__bases__)