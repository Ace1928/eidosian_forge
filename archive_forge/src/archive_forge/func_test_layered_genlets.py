from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_layered_genlets(self):
    seen = []
    for ii in gr2(5, seen):
        seen.append(ii)
    self.assertEqual(seen, [1, 1, 2, 4, 3, 9, 4, 16])