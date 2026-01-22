import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getSignatureString_w_kwargs(self):
    method = self._makeOne()
    method.kwargs = 'kw'
    self.assertEqual(method.getSignatureString(), '(**kw)')