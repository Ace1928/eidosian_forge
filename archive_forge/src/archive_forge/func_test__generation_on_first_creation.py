import unittest
from zope.interface.tests import OptimizationTestMixin
def test__generation_on_first_creation(self):
    registry = self._makeOne()
    self.assertEqual(registry._generation, 1)