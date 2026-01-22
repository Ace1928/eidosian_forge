import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_get_hit_w__v_attrs(self):
    spec = self._makeOne()
    foo = object()
    spec._v_attrs = {'foo': foo}
    self.assertTrue(spec.get('foo') is foo)