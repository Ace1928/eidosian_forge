import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__providedBy_alsoProvides(self):
    from zope.interface.declarations import alsoProvides
    from zope.interface.declarations import implementer
    from zope.interface.declarations import providedBy
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    alsoProvides(foo, IBar)
    inst = providedBy(foo)
    self.assertEqual(repr(inst), 'directlyProvides(Foo, IBar, classImplements(Foo, IFoo))')