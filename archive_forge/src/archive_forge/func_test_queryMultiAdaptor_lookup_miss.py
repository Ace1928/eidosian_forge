import unittest
from zope.interface.tests import OptimizationTestMixin
def test_queryMultiAdaptor_lookup_miss(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    registry = self._makeRegistry()
    subr = self._makeSubregistry()
    subr._adapters = [{}, {}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    alb.lookup = alb._uncached_lookup
    subr._v_lookup = alb
    _default = object()
    result = alb.queryMultiAdapter((foo,), IBar, default=_default)
    self.assertIs(result, _default)