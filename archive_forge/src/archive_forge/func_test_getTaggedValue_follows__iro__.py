import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getTaggedValue_follows__iro__(self):
    from zope.interface import Interface
    class_A = self._make_taggedValue_tree(object)
    self.assertEqual(class_A.tag.__name__, 'C')
    iface_A = self._make_taggedValue_tree(Interface)
    self.assertEqual(iface_A['tag'].__name__, 'C')
    self.assertEqual(iface_A.getTaggedValue('tag'), 'C')
    self.assertEqual(iface_A.queryTaggedValue('tag'), 'C')
    assert iface_A.__bases__[0].__name__ == 'B'
    iface_A.__bases__[0].setTaggedValue('tag', 'B')
    self.assertEqual(iface_A.getTaggedValue('tag'), 'B')