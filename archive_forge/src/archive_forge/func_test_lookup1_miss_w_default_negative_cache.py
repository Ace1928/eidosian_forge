import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup1_miss_w_default_negative_cache(self):
    _called_with = []
    _default = object()

    def _lookup(self, required, provided, name):
        _called_with.append((required, provided, name))
    lb = self._makeOne(uc_lookup=_lookup)
    found = lb.lookup1('A', 'B', 'C', _default)
    self.assertIs(found, _default)
    found = lb.lookup1('A', 'B', 'C', _default)
    self.assertIs(found, _default)
    self.assertEqual(_called_with, [(('A',), 'B', 'C')])