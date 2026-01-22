from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def test_theta_est(self):
    objval, thetavals = self.pest.theta_est()
    self.assertAlmostEqual(thetavals['k1'], 5.0 / 6.0, places=4)
    self.assertAlmostEqual(thetavals['k2'], 5.0 / 3.0, places=4)
    self.assertAlmostEqual(thetavals['k3'], 1.0 / 6000.0, places=7)