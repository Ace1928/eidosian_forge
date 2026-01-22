import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_accessed_via_inst_w_provides(self):
    from zope.interface.declarations import Provides
    from zope.interface.declarations import directlyProvides
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    IBaz = InterfaceClass('IBaz')

    @implementer(IFoo)
    class Foo:
        pass
    Foo.__provides__ = Provides(Foo, IBar)
    Foo.__providedBy__ = self._makeOne()
    foo = Foo()
    directlyProvides(foo, IBaz)
    self.assertEqual(list(foo.__providedBy__), [IBaz, IFoo])