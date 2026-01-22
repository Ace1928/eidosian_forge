import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_lookup_repeated_hit(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    _expected = object()
    subr._adapters = [{}, {IFoo: {IBar: {'': _expected}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_lookup((IFoo,), IBar)
    result2 = alb._uncached_lookup((IFoo,), IBar)
    self.assertIs(result, _expected)
    self.assertIs(result2, _expected)