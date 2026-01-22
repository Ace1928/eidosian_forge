import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getUtilitiesFor_hit(self):
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
    self.assertEqual(sorted(comp.getUtilitiesFor(ifoo)), [(_name1, _to_reg), (_name2, _to_reg)])