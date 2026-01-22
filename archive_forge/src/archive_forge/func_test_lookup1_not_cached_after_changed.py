import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup1_not_cached_after_changed(self):
    _called_with = []
    a, b, c = (object(), object(), object())
    _results = [a, b, c]

    def _lookup(self, required, provided, name):
        _called_with.append((required, provided, name))
        return _results.pop(0)
    lb = self._makeOne(uc_lookup=_lookup)
    found = lb.lookup1('A', 'B', 'C')
    lb.changed(lb)
    found = lb.lookup1('A', 'B', 'C')
    self.assertIs(found, b)
    self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
    self.assertEqual(_results, [c])