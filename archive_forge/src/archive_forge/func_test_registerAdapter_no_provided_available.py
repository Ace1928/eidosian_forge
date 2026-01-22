import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_no_provided_available(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ibar = IFoo('IBar')
    _info = 'info'
    _name = 'name'

    class _Factory:
        pass
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.registerAdapter, _Factory, (ibar,), name=_name, info=_info)