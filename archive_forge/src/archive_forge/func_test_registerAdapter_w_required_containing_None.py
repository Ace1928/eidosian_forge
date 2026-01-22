import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_w_required_containing_None(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interface import Interface
    from zope.interface.interfaces import Registered
    from zope.interface.registry import AdapterRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _name = 'name'

    class _Factory:
        pass
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerAdapter(_Factory, [None], provided=ifoo, name=_name, info=_info)
    self.assertTrue(comp.adapters._adapters[1][Interface][ifoo][_name] is _Factory)
    self.assertEqual(comp._adapter_registrations[(Interface,), ifoo, _name], (_Factory, _info))
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, AdapterRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (Interface,))
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _Factory)