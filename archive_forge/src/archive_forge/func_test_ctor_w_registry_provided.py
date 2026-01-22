import unittest
from zope.interface.tests import OptimizationTestMixin
def test_ctor_w_registry_provided(self):
    from zope.interface import Interface
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    alb = self._makeOne(registry)
    self.assertEqual(sorted(alb._extendors.keys()), sorted([IBar, IFoo, Interface]))
    self.assertEqual(alb._extendors[IFoo], [IFoo, IBar])
    self.assertEqual(alb._extendors[IBar], [IBar])
    self.assertEqual(sorted(alb._extendors[Interface]), sorted([IFoo, IBar]))