import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_get_residual(self):
    dat = uidata.UIData(self.m)
    assert rpt.get_residual(dat, self.m.c3) is None
    dat.calculate_constraints()
    self.assertAlmostEqual(rpt.get_residual(dat, self.m.c3), 2.0)
    assert rpt.get_residual(dat, self.m.c8) == 'Divide_by_0'
    assert rpt.get_residual(dat, self.m.c8b) == 'Divide_by_0'
    self.m.x[2] = 0
    assert rpt.get_residual(dat, self.m.c4) == 'Divide_by_0'
    self.m.x[2] = 2
    assert rpt.get_residual(dat, self.m.c4) == 'Divide_by_0'
    dat.calculate_constraints()
    self.assertAlmostEqual(rpt.get_residual(dat, self.m.c4), 3.0 / 2.0)
    self.assertAlmostEqual(rpt.get_residual(dat, self.m.c9), 0)
    self.assertAlmostEqual(rpt.get_residual(dat, self.m.c10), 0)