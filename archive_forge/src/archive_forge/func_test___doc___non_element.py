import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___doc___non_element(self):
    from zope.interface import Interface

    class IHaveADocString(Interface):
        """xxx"""
    self.assertEqual(IHaveADocString.__doc__, 'xxx')
    self.assertEqual(list(IHaveADocString), [])