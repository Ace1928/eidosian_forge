import unittest
from zope.interface.tests import OptimizationTestMixin
def test__setBases_wo_stray_entry(self):
    before = self._makeOne()
    stray = self._makeOne()
    after = self._makeOne()
    sub = self._makeOne([before])
    sub.__dict__['__bases__'].append(stray)
    sub.__bases__ = [after]
    self.assertEqual(len(before._v_subregistries), 0)
    self.assertEqual(len(after._v_subregistries), 1)
    self.assertIn(sub, after._v_subregistries)