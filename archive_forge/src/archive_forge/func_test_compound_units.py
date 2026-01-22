from .. import units as pq
from .common import TestCase
def test_compound_units(self):
    pc_per_cc = pq.CompoundUnit('pc/cm**3')
    self.assertEqual(str(pc_per_cc.dimensionality), '(pc/cm**3)')
    self.assertEqual(str(pc_per_cc), '1 (pc/cm**3)')
    temp = pc_per_cc * pq.CompoundUnit('m/m**3')
    self.assertEqual(str(temp.dimensionality), '(pc/cm**3)*(m/m**3)')
    self.assertEqual(str(temp), '1.0 (pc/cm**3)*(m/m**3)')