import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_existing_attr_as_Implements(self):
    from zope.interface.declarations import Implements
    impl = Implements()

    class Foo:
        __implemented__ = impl
    self.assertTrue(self._callFUT(Foo) is impl)