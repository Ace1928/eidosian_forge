import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_providedBy_valid_spec(self):
    from zope.interface.declarations import Provides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    foo = Foo()
    foo.__providedBy__ = Provides(Foo, IFoo)
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [IFoo])