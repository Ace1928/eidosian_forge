import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getAllUtilitiesRegisteredFor_hit(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _name1 = 'name1'
    _name2 = 'name2'
    _to_reg = object()
    comp = self._makeOne()
    comp.registerUtility(_to_reg, ifoo, name=_name1)
    comp.registerUtility(_to_reg, ifoo, name=_name2)
    self.assertEqual(list(comp.getAllUtilitiesRegisteredFor(ifoo)), [_to_reg])