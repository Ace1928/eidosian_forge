import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getDirectTaggedValue_ignores__iro__(self):
    from zope.interface import Interface
    A = self._make_taggedValue_tree(Interface)
    self.assertIsNone(A.queryDirectTaggedValue('tag'))
    self.assertEqual([], list(A.getDirectTaggedValueTags()))
    with self.assertRaises(KeyError):
        A.getDirectTaggedValue('tag')
    A.setTaggedValue('tag', 'A')
    self.assertEqual(A.queryDirectTaggedValue('tag'), 'A')
    self.assertEqual(A.getDirectTaggedValue('tag'), 'A')
    self.assertEqual(['tag'], list(A.getDirectTaggedValueTags()))
    assert A.__bases__[1].__name__ == 'C'
    C = A.__bases__[1]
    self.assertEqual(C.queryDirectTaggedValue('tag'), 'C')
    self.assertEqual(C.getDirectTaggedValue('tag'), 'C')
    self.assertEqual(['tag'], list(C.getDirectTaggedValueTags()))