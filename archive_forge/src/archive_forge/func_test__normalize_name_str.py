import unittest
from zope.interface.tests import OptimizationTestMixin
def test__normalize_name_str(self):
    from zope.interface.adapter import _normalize_name
    STR = b'str'
    UNICODE = 'str'
    norm = _normalize_name(STR)
    self.assertEqual(norm, UNICODE)
    self.assertIsInstance(norm, type(UNICODE))