import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_subscribe_again(self):
    spec = self._makeOne()
    dep = DummyDependent()
    spec.subscribe(dep)
    spec.subscribe(dep)
    self.assertEqual(spec.dependents[dep], 2)