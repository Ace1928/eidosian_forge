import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_changed_w_existing__v_attrs(self):
    decl = self._getEmpty()
    decl._v_attrs = object()
    decl.changed(decl)
    self.assertFalse(decl._v_attrs)