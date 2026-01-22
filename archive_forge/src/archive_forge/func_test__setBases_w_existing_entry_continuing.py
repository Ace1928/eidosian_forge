import unittest
from zope.interface.tests import OptimizationTestMixin
def test__setBases_w_existing_entry_continuing(self):
    before = self._makeOne()
    after = self._makeOne()
    sub = self._makeOne([before])
    sub.__bases__ = [before, after]
    self.assertEqual(len(before._v_subregistries), 1)
    self.assertEqual(len(after._v_subregistries), 1)
    self.assertIn(sub, before._v_subregistries)
    self.assertIn(sub, after._v_subregistries)