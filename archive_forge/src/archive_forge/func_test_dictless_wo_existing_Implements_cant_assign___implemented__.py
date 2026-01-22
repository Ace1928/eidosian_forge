import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_dictless_wo_existing_Implements_cant_assign___implemented__(self):

    class Foo:

        def _get_impl(self):
            raise NotImplementedError()

        def _set_impl(self, val):
            raise TypeError
        __implemented__ = property(_get_impl, _set_impl)

        def __call__(self):
            raise NotImplementedError()
    foo = Foo()
    self.assertRaises(TypeError, self._callFUT, foo)