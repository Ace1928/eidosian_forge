import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test__repr__w_interface(self):
    method = self._makeOne()
    method.kwargs = 'kw'
    method.interface = type(self)
    r = repr(method)
    self.assertTrue(r.startswith('<zope.interface.interface.Method object at'), r)
    self.assertTrue(r.endswith(' ' + __name__ + '.MethodTests.TestMethod(**kw)>'), r)