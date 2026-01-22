import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_v_attrs(self):
    decl = self._getEmpty()
    self.assertEqual(decl._v_attrs, {})
    decl._v_attrs['attr'] = 42
    self.assertEqual(decl._v_attrs, {})
    self.assertIsNone(decl.get('attr'))
    attrs = decl._v_attrs = {}
    attrs['attr'] = 42
    self.assertEqual(decl._v_attrs, {})
    self.assertIsNone(decl.get('attr'))