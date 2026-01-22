import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_var_get_bounds(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    self.m.x[1].setlb(0)
    self.m.x[1].setub(10)
    self.assertAlmostEqual(cdi.get('lb'), 0)
    self.assertAlmostEqual(cdi.get('ub'), 10)