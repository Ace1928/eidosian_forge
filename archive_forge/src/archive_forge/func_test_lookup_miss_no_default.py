import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup_miss_no_default(self):
    _called_with = []

    def _lookup(self, required, provided, name):
        _called_with.append((required, provided, name))
    lb = self._makeOne(uc_lookup=_lookup)
    found = lb.lookup(('A',), 'B', 'C')
    self.assertIsNone(found)
    self.assertEqual(_called_with, [(('A',), 'B', 'C')])