import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def test_mc_2d(self):
    m = ConcreteModel()
    m.x = Var(bounds=(pi / 6, pi / 3), initialize=pi / 4)
    m.e = Expression(expr=cos(pow(m.x, 2)) * sin(pow(m.x, -3)))
    mc_ccVals, mc_cvVals, aff_cc, aff_cv = make2dPlot(m.e.expr, 50)
    self.assertAlmostEqual(mc_ccVals[1], 0.6443888590411435)
    self.assertAlmostEqual(mc_cvVals[1], 0.2328315489072924)
    self.assertAlmostEqual(aff_cc[1], 0.9674274332870583)
    self.assertAlmostEqual(aff_cv[1], -1.578938503009686)