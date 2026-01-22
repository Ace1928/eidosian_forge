import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_same_class_via_class(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    cpbp = Foo.__provides__ = self._makeOne(Foo, IFoo)
    self.assertTrue(Foo.__provides__ is cpbp)