import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___contains__w_unrelated_iface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    decl = self._makeOne()
    self.assertNotIn(IFoo, decl)