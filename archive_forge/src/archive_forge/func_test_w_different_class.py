import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_different_class(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass

    class Bar(Foo):
        pass
    bar = Bar()
    Foo.__provides__ = self._makeOne(Foo, IFoo)
    self.assertRaises(AttributeError, getattr, Bar, '__provides__')
    self.assertRaises(AttributeError, getattr, bar, '__provides__')