import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getAdapters_non_empty(self):
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

    class _Factory1:

        def __init__(self, context1, context2):
            self.context = (context1, context2)

    class _Factory2:

        def __init__(self, context1, context2):
            self.context = (context1, context2)
    _name1 = 'name1'
    _name2 = 'name2'
    comp = self._makeOne()
    comp.registerAdapter(_Factory1, (ibar, ibaz), ifoo, name=_name1)
    comp.registerAdapter(_Factory2, (ibar, ibaz), ifoo, name=_name2)
    found = sorted(comp.getAdapters((_context1, _context2), ifoo))
    self.assertEqual(len(found), 2)
    self.assertEqual(found[0][0], _name1)
    self.assertTrue(isinstance(found[0][1], _Factory1))
    self.assertEqual(found[1][0], _name2)
    self.assertTrue(isinstance(found[1][1], _Factory2))