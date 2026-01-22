import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_existing_provides_hit(self):
    from zope.interface.declarations import directlyProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    obj = Foo()
    directlyProvides(obj, IFoo)
    self._callFUT(obj, IFoo)
    self.assertEqual(list(obj.__provides__), [])