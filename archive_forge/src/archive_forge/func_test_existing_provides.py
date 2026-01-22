import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_existing_provides(self):
    from zope.interface.declarations import directlyProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    foo = Foo()
    directlyProvides(foo, IFoo)
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [IFoo])