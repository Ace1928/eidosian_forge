import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_no_cached_spec(self):
    from zope.interface import declarations
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    cache = {}

    class Foo:
        pass
    with _Monkey(declarations, InstanceDeclarations=cache):
        spec = self._callFUT(Foo, IFoo)
    self.assertEqual(list(spec), [IFoo])
    self.assertTrue(cache[Foo, IFoo] is spec)