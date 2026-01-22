import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_miss(self):
    spec = self._makeOne()
    dep = DummyDependent()
    self.assertRaises(KeyError, spec.unsubscribe, dep)