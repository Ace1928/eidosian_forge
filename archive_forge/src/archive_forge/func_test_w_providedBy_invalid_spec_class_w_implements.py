import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_providedBy_invalid_spec_class_w_implements(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    foo.__providedBy__ = object()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [IFoo])