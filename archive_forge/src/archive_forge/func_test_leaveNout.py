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
def test_leaveNout(self):
    lNo_theta = self.pest.theta_est_leaveNout(1)
    self.assertTrue(lNo_theta.shape == (6, 2))
    results = self.pest.leaveNout_bootstrap_test(1, None, 3, 'Rect', [0.5, 1.0], seed=5436)
    self.assertTrue(len(results) == 6)
    i = 1
    samples = results[i][0]
    lno_theta = results[i][1]
    bootstrap_theta = results[i][2]
    self.assertTrue(samples == [1])
    self.assertTrue(lno_theta.shape[0] == 1)
    self.assertTrue(set(lno_theta.columns) >= set([0.5, 1.0]))
    self.assertTrue(lno_theta[1.0].sum() == 1)
    self.assertTrue(bootstrap_theta.shape[0] == 3)
    self.assertTrue(bootstrap_theta[1.0].sum() == 3)