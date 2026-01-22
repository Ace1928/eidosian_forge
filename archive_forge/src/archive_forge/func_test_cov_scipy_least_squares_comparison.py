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
def test_cov_scipy_least_squares_comparison(self):
    """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

    def model(theta, t):
        """
            Model to be fitted y = model(theta, t)
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]

            Returns:
                y: model predictions [need to check paper for units]
            """
        asymptote = theta[0]
        rate_constant = theta[1]
        return asymptote * (1 - np.exp(-rate_constant * t))

    def residual(theta, t, y):
        """
            Calculate residuals
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]
                y: dependent variable [?]
            """
        return y - model(theta, t)
    t = self.data['hour'].to_numpy()
    y = self.data['y'].to_numpy()
    theta_guess = np.array([15, 0.5])
    sol = scipy.optimize.least_squares(residual, theta_guess, method='trf', args=(t, y), verbose=2)
    theta_hat = sol.x
    self.assertAlmostEqual(theta_hat[0], 19.1426, places=2)
    self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)
    r = residual(theta_hat, t, y)
    sigre = np.matmul(r.T, r / (len(y) - 2))
    cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))
    self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)
    self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)
    self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)
    self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)