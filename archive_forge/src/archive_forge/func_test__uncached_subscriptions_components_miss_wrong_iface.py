import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_subscriptions_components_miss_wrong_iface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    IQux = InterfaceClass('IQux')
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    irrelevant = object()
    subr._subscribers = [{}, {IFoo: {IQux: {'': irrelevant}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_subscriptions((IFoo,), IBar)
    self.assertEqual(result, [])