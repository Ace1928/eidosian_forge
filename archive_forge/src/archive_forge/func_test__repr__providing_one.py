import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__providing_one(self):
    from zope.interface import Interface

    class IFoo(Interface):
        """Does nothing"""
    inst = self._makeOne(type(self), type, IFoo)
    self.assertEqual(repr(inst), 'directlyProvides(TestClassProvidesRepr, IFoo)')