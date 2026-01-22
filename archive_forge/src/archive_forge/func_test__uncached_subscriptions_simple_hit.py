import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_subscriptions_simple_hit(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()

    class Foo:

        def __lt__(self, other):
            return True
    _exp1, _exp2 = (Foo(), Foo())
    subr._subscribers = [{}, {IFoo: {IBar: {'': (_exp1, _exp2)}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_subscriptions((IFoo,), IBar)
    self.assertEqual(sorted(result), sorted([_exp1, _exp2]))