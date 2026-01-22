import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_nonclass_can_assign_attr(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    foo = Foo()
    decorator = self._makeOne(IFoo)
    returned = decorator(foo)
    self.assertTrue(returned is foo)
    spec = foo.__implemented__
    self.assertEqual(spec.__name__, 'zope.interface.tests.test_declarations.?')
    self.assertIsNone(spec.inherit)
    self.assertIs(foo.__implemented__, spec)