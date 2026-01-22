import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterHandler_wo_explicit_required(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import HandlerRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')

    class _Factory:
        __component_adapts__ = (ifoo,)
    comp = self._makeOne()
    comp.registerHandler(_Factory)
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterHandler(_Factory)
    self.assertTrue(unreg)
    self.assertEqual(len(_events), 1)
    args, kw = _events[0]
    event, = args
    self.assertEqual(kw, {})
    self.assertTrue(isinstance(event, Unregistered))
    self.assertTrue(isinstance(event.object, HandlerRegistration))
    self.assertTrue(event.object.registry is comp)
    self.assertEqual(event.object.required, (ifoo,))
    self.assertEqual(event.object.name, '')
    self.assertEqual(event.object.info, '')
    self.assertTrue(event.object.factory is _Factory)