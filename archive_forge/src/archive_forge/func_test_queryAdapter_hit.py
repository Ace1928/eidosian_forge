import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryAdapter_hit(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')

    class _Factory:

        def __init__(self, context):
            self.context = context

    @implementer(ibar)
    class _Context:
        pass
    _context = _Context()
    comp = self._makeOne()
    comp.registerAdapter(_Factory, (ibar,), ifoo)
    adapter = comp.queryAdapter(_context, ifoo)
    self.assertTrue(isinstance(adapter, _Factory))
    self.assertTrue(adapter.context is _context)