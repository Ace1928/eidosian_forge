import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__implementedBy_generic_callable(self):
    from zope.interface.declarations import implementedBy

    class Callable:

        def __call__(self):
            return self
    inst = implementedBy(Callable())
    self.assertEqual(repr(inst), 'classImplements({}.?)'.format(__name__))
    c = Callable()
    c.__name__ = 'Callable'
    inst = implementedBy(c)
    self.assertEqual(repr(inst), 'classImplements(Callable)')