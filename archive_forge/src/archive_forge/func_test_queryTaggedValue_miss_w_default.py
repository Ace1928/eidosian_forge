import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_queryTaggedValue_miss_w_default(self):
    element = self._makeOne()
    self.assertEqual(element.queryTaggedValue('nonesuch', 'bar'), 'bar')