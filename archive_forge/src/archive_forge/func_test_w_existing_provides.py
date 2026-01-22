import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_existing_provides(self):
    from zope.interface.declarations import ProvidesClass
    from zope.interface.declarations import directlyProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')

    class Foo:
        pass
    obj = Foo()
    directlyProvides(obj, IFoo)
    self._callFUT(obj, IBar)
    self.assertIsInstance(obj.__provides__, ProvidesClass)
    self.assertEqual(list(obj.__provides__), [IFoo, IBar])