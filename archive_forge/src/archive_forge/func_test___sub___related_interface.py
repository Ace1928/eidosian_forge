import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___sub___related_interface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    before = self._makeOne(IFoo)
    after = before - IFoo
    self.assertEqual(list(after), [])