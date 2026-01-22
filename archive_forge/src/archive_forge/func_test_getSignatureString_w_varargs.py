import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getSignatureString_w_varargs(self):
    method = self._makeOne()
    method.varargs = 'args'
    self.assertEqual(method.getSignatureString(), '(*args)')