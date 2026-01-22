import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_normal_object(self):
    from zope.interface.declarations import ProvidesClass
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    obj = Foo()
    self._callFUT(obj, IFoo)
    self.assertIsInstance(obj.__provides__, ProvidesClass)
    self.assertEqual(list(obj.__provides__), [IFoo])