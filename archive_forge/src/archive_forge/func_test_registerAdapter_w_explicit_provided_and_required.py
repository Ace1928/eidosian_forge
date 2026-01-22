import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_w_explicit_provided_and_required(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Registered
    from zope.interface.registry import AdapterRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _info = 'info'
    _name = 'name'

    def _factory(context):
        raise NotImplementedError()
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerAdapter(_factory, (ibar,), ifoo, _name, _info)
    self.assertTrue(comp.adapters._adapters[1][ibar][ifoo][_name] is _factory)
    self.assertEqual(comp._adapter_registrations[(ibar,), ifoo, _name], (_factory, _info))
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, AdapterRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (ibar,))
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _factory)