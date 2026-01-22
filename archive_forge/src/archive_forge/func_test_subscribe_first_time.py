import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_subscribe_first_time(self):
    spec = self._makeOne()
    dep = DummyDependent()
    spec.subscribe(dep)
    self.assertEqual(len(spec.dependents), 1)
    self.assertEqual(spec.dependents[dep], 1)