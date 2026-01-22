import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterHandler_hit_w_only_explicit_provided(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.interfaces import Unregistered
    from zope.interface.registry import HandlerRegistration

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()

    def _factory(context):
        raise NotImplementedError()
    comp = self._makeOne()
    comp.registerHandler(_factory, (ifoo,))
    _monkey, _events = self._wrapEvents()
    with _monkey:
        unreg = comp.unregisterHandler(required=(ifoo,))
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
    self.assertTrue(event.object.factory is None)