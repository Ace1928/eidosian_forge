import unittest
from zope.interface.tests import OptimizationTestMixin
def test__normalize_name_unicode(self):
    from zope.interface.adapter import _normalize_name
    USTR = 'ustr'
    self.assertEqual(_normalize_name(USTR), USTR)