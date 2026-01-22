import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerAdapter_w_required_containing_class(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementedBy
    from zope.interface.declarations import implementer
    from zope.interface.interfaces import Registered
    from zope.interface.registry import AdapterRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    _info = 'info'
    _name = 'name'

    class _Factory:
        pass

    @implementer(ibar)
    class _Context:
        pass
    _ctx_impl = implementedBy(_Context)
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerAdapter(_Factory, [_Context], provided=ifoo, name=_name, info=_info)
    self.assertTrue(comp.adapters._adapters[1][_ctx_impl][ifoo][_name] is _Factory)
    self.assertEqual(comp._adapter_registrations[(_ctx_impl,), ifoo, _name], (_Factory, _info))
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Registered))
    self.assertTrue(isinstance(event.object, AdapterRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertTrue(event.object.provided is ifoo)
    self.assertEqual(event.object.required, (_ctx_impl,))
    self.assertTrue(event.object.name is _name)
    self.assertTrue(event.object.info is _info)
    self.assertTrue(event.object.factory is _Factory)