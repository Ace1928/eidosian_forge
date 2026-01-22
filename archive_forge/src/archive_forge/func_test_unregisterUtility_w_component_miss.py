import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterUtility_w_component_miss(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _name = 'name'
    _to_reg = object()
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterUtility(_to_reg, ifoo, _name)
    self.assertFalse(unreg)
    self.assertFalse(_events)