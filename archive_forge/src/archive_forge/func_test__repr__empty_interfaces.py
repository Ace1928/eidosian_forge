import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__empty_interfaces(self):
    inst = self._makeOne(type(self))
    self.assertEqual(repr(inst), 'directlyProvides(TestProvidesClassRepr)')