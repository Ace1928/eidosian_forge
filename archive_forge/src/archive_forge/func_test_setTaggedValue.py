import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_setTaggedValue(self):
    element = self._makeOne()
    element.setTaggedValue('foo', 'bar')
    self.assertEqual(list(element.getTaggedValueTags()), ['foo'])
    self.assertEqual(element.getTaggedValue('foo'), 'bar')
    self.assertEqual(element.queryTaggedValue('foo'), 'bar')