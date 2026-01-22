import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getUtility_miss(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import ComponentLookupError

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()
    self.assertRaises(ComponentLookupError, comp.getUtility, ifoo)