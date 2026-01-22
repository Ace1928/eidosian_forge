import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerHandler_wo_explicit_required_no_event(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _info = 'info'
    _blank = ''

    class _Factory:
        __component_adapts__ = (ifoo,)
        pass
    comp = self._makeOne()
    _monkey, _events = self._wrapEvents()
    with _monkey:
        comp.registerHandler(_Factory, info=_info, event=False)
    reg = comp.adapters._subscribers[1][ifoo][None][_blank]
    self.assertEqual(len(reg), 1)
    self.assertTrue(reg[0] is _Factory)
    self.assertEqual(comp._handler_registrations, [((ifoo,), _blank, _Factory, _info)])
    self.assertEqual(len(_events), 0)