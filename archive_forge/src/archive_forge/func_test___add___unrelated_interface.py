import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___add___unrelated_interface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    before = self._makeOne(IFoo)
    after = before + IBar
    self.assertIsInstance(after, self._getTargetClass())
    self.assertEqual(list(after), [IFoo, IBar])