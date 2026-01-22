import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def test_improved_bounds(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 100), initialize=5)
    improved_bounds = ComponentMap()
    improved_bounds[m.x] = (10, 20)
    mc_expr = mc(m.x, improved_var_bounds=improved_bounds)
    self.assertEqual(mc_expr.lower(), 10)
    self.assertEqual(mc_expr.upper(), 20)