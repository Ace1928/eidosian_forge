import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_case_shortnamelabeler_overflow(self):
    m = self.m
    lbl = ShortNameLabeler(4, '_', caseInsensitive=True)
    for i in range(9):
        self.assertEqual(lbl(m.mycomp), 'p_%d_' % (i + 1))
    with self.assertRaisesRegex(RuntimeError, 'Too many identifiers'):
        lbl(m.mycomp)