import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_handle_non_empty(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import implementer

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    _called_1 = []

    def _factory_1(context):
        _called_1.append(context)
    _called_2 = []

    def _factory_2(context):
        _called_2.append(context)
    comp = self._makeOne()
    comp.registerHandler(_factory_1, (ifoo,))
    comp.registerHandler(_factory_2, (ifoo,))

    @implementer(ifoo)
    class Bar:
        pass
    bar = Bar()
    comp.handle(bar)
    self.assertEqual(_called_1, [bar])
    self.assertEqual(_called_2, [bar])