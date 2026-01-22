import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___get___class(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    spec = self._makeOne(Foo, IFoo)
    Foo.__provides__ = spec
    self.assertIs(Foo.__provides__, spec)