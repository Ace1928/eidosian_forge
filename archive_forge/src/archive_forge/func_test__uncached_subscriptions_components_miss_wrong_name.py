import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_subscriptions_components_miss_wrong_name(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    wrongname = object()
    subr._subscribers = [{}, {IFoo: {IBar: {'wrongname': wrongname}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_subscriptions((IFoo,), IBar)
    self.assertEqual(result, [])