import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def test_fraction_to_the_boundary_helper_ub(self):
    tau = 0.9
    x = np.array([0, 0, 0, 0], dtype=np.double)
    xu = np.array([np.inf, 1, np.inf, 1], dtype=np.double)
    delta_x = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 1)
    delta_x = np.array([1, 1, 1, 1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 0.9)
    delta_x = np.array([10, 10, 10, 10], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 0.09)
    delta_x = np.array([-1, -1, -1, -1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 1)
    delta_x = np.array([10, -1, 10, -1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 1)
    delta_x = np.array([10, 1, 10, 1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 0.9)
    delta_x = np.array([-1, 10, -1, 1], dtype=np.double)
    alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
    self.assertAlmostEqual(alpha, 0.09)