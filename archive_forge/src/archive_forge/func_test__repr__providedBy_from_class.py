import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__providedBy_from_class(self):
    from zope.interface.declarations import implementer
    from zope.interface.declarations import providedBy
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    @implementer(IFoo)
    class Foo:
        pass
    inst = providedBy(Foo())
    self.assertEqual(repr(inst), 'classImplements(Foo, IFoo)')