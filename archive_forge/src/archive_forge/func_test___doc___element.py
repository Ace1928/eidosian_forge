import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___doc___element(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class IDocstring(Interface):
        """xxx"""
    self.assertEqual(IDocstring.__doc__, 'xxx')
    self.assertEqual(list(IDocstring), [])

    class IDocstringAndAttribute(Interface):
        """xxx"""
        __doc__ = Attribute('the doc')
    self.assertEqual(IDocstringAndAttribute.__doc__, '')
    self.assertEqual(list(IDocstringAndAttribute), ['__doc__'])