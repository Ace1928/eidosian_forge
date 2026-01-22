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
@unittest.skipIf(not pynumero_ASL_available, 'pynumero ASL is not available')
@unittest.skipIf(not parmest.inverse_reduced_hessian_available, 'Cannot test covariance matrix: required ASL dependency is missing')
def test_theta_est_cov(self):
    objval, thetavals, cov = self.pest.theta_est(calc_cov=True, cov_n=6)
    self.assertAlmostEqual(objval, 4.3317112, places=2)
    self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2)
    self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2)
    self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
    self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
    self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
    self.assertAlmostEqual(cov.iloc[1, 1], 0.04124, places=2)
    ' Why does the covariance matrix from parmest not match the paper? Parmest is\n        calculating the exact reduced Hessian. The paper (Rooney and Bielger, 2001) likely\n        employed the first order approximation common for nonlinear regression. The paper\n        values were verified with Scipy, which uses the same first order approximation.\n        The formula used in parmest was verified against equations (7-5-15) and (7-5-16) in\n        "Nonlinear Parameter Estimation", Y. Bard, 1974.\n        '