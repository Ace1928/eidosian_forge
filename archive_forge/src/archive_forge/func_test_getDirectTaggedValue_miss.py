import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getDirectTaggedValue_miss(self):
    element = self._makeOne()
    self.assertRaises(KeyError, element.getDirectTaggedValue, 'nonesuch')