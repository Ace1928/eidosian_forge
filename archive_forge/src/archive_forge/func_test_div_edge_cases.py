import math
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException
import pyomo.contrib.fbbt.interval as interval
def test_div_edge_cases(self):
    lb, ub = self.div(0, -1e-16, 0, 0, 1e-08)
    self.assertEqual(lb, -interval.inf)
    self.assertEqual(ub, interval.inf)
    lb, ub = self.div(0, 1e-16, 0, 0, 1e-08)
    self.assertEqual(lb, -interval.inf)
    self.assertEqual(ub, interval.inf)
    lb, ub = self.div(-1e-16, 0, 0, 0, 1e-08)
    self.assertEqual(lb, -interval.inf)
    self.assertEqual(ub, interval.inf)
    lb, ub = self.div(1e-16, 0, 0, 0, 1e-08)
    self.assertEqual(lb, -interval.inf)
    self.assertEqual(ub, interval.inf)