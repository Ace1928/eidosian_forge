import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_declarations_in_instance_but_not_class(self):
    from zope.interface.declarations import directlyProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    foo = Foo()
    directlyProvides(foo, IFoo)
    self.assertEqual(list(self._callFUT(foo)), [IFoo])