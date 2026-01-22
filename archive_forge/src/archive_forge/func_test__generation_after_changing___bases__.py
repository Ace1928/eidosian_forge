import unittest
from zope.interface.tests import OptimizationTestMixin
def test__generation_after_changing___bases__(self):

    class _Base:
        pass
    registry = self._makeOne()
    registry.__bases__ = (_Base,)
    self.assertEqual(registry._generation, 2)