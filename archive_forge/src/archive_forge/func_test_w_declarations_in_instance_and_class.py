import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_declarations_in_instance_and_class(self):
    from zope.interface.declarations import directlyProvides
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    directlyProvides(foo, IBar)
    self.assertEqual(list(self._callFUT(foo)), [IBar])