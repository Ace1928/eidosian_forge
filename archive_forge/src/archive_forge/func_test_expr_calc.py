import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_expr_calc(self):
    cdi = ComponentDataItem(parent=None, ui_data=UIData(model=self.m), o=self.m.b1.e1)
    cdi.ui_data.calculate_expressions()
    self.assertAlmostEqual(cdi.get('value'), 3)
    self.assertIsInstance(cdi.get('expr'), str)