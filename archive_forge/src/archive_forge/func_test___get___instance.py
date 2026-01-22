import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___get___instance(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    spec = self._makeOne(Foo, IFoo)
    Foo.__provides__ = spec

    def _test():
        foo = Foo()
        return foo.__provides__
    self.assertRaises(AttributeError, _test)