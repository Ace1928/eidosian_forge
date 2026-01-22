import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredAdapters_notempty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.registry import AdapterRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IFoo')
    _info = 'info'
    _name1 = 'name1'
    _name2 = 'name2'

    class _Factory:
        pass
    comp = self._makeOne()
    comp.registerAdapter(_Factory, (ibar,), ifoo, _name1, _info)
    comp.registerAdapter(_Factory, (ibar,), ifoo, _name2, _info)
    reg = sorted(comp.registeredAdapters(), key=lambda r: r.name)
    self.assertEqual(len(reg), 2)
    self.assertTrue(isinstance(reg[0], AdapterRegistration))
    self.assertTrue(reg[0].registry is comp)
    self.assertTrue(reg[0].provided is ifoo)
    self.assertEqual(reg[0].required, (ibar,))
    self.assertTrue(reg[0].name is _name1)
    self.assertTrue(reg[0].info is _info)
    self.assertTrue(reg[0].factory is _Factory)
    self.assertTrue(isinstance(reg[1], AdapterRegistration))
    self.assertTrue(reg[1].registry is comp)
    self.assertTrue(reg[1].provided is ifoo)
    self.assertEqual(reg[1].required, (ibar,))
    self.assertTrue(reg[1].name is _name2)
    self.assertTrue(reg[1].info is _info)
    self.assertTrue(reg[1].factory is _Factory)