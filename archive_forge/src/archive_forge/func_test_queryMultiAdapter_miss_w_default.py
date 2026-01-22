import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_queryMultiAdapter_miss_w_default(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    ibar = IFoo('IBar')
    ibaz = IFoo('IBaz')

    @implementer(ibar)
    class _Context1:
        pass

    @implementer(ibaz)
    class _Context2:
        pass
    _context1 = _Context1()
    _context2 = _Context2()
    _default = object()
    comp = self._makeOne()
    self.assertTrue(comp.queryMultiAdapter((_context1, _context2), ifoo, default=_default) is _default)