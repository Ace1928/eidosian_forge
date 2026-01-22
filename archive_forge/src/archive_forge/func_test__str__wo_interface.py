import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test__str__wo_interface(self):
    method = self._makeOne()
    method.kwargs = 'kw'
    r = str(method)
    self.assertEqual(r, 'TestMethod(**kw)')