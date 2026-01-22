import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getDirectTaggedValueTags(self):
    element = self._makeOne()
    self.assertEqual([], list(element.getDirectTaggedValueTags()))
    element.setTaggedValue('foo', 'bar')
    self.assertEqual(['foo'], list(element.getDirectTaggedValueTags()))