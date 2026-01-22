import unittest
from zope.interface.tests import OptimizationTestMixin
def test_ctor_no_bases(self):
    ar = self._makeOne()
    self.assertEqual(len(ar._v_subregistries), 0)