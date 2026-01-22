import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_active_equalities(self):
    eq = [self.m.c1, self.m.c2, self.m.c3, self.m.c4, self.m.c5, self.m.c6, self.m.c7]
    for i, o in enumerate(rpt.active_equalities(self.m)):
        assert o == eq[i]