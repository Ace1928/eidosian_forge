import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def test_nonpyomo_numeric(self):
    mc_expr = mc(-2)
    self.assertEqual(mc_expr.lower(), -2)
    self.assertEqual(mc_expr.upper(), -2)