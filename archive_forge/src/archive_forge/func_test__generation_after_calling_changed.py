import unittest
from zope.interface.tests import OptimizationTestMixin
def test__generation_after_calling_changed(self):
    registry = self._makeOne()
    orig = object()
    registry.changed(orig)
    self.assertEqual(registry._generation, 2)
    self.assertEqual(registry._v_lookup._changed, (registry, orig))