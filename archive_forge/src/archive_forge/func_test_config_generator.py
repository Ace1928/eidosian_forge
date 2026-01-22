import logging
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.contrib.trustregion.TRF import trust_region_method, _trf_config
def test_config_generator(self):
    CONFIG = _trf_config()
    self.assertEqual(CONFIG.solver, 'ipopt')
    self.assertFalse(CONFIG.keepfiles)
    self.assertFalse(CONFIG.tee)
    self.assertFalse(CONFIG.verbose)
    self.assertEqual(CONFIG.trust_radius, 1.0)
    self.assertEqual(CONFIG.minimum_radius, 1e-06)
    self.assertEqual(CONFIG.maximum_radius, 100.0)
    self.assertEqual(CONFIG.maximum_iterations, 50)
    self.assertEqual(CONFIG.feasibility_termination, 1e-05)
    self.assertEqual(CONFIG.step_size_termination, 1e-05)
    self.assertEqual(CONFIG.minimum_feasibility, 0.0001)
    self.assertEqual(CONFIG.switch_condition_kappa_theta, 0.1)
    self.assertEqual(CONFIG.switch_condition_gamma_s, 2.0)
    self.assertEqual(CONFIG.radius_update_param_gamma_c, 0.5)
    self.assertEqual(CONFIG.radius_update_param_gamma_e, 2.5)
    self.assertEqual(CONFIG.ratio_test_param_eta_1, 0.05)
    self.assertEqual(CONFIG.ratio_test_param_eta_2, 0.2)
    self.assertEqual(CONFIG.maximum_feasibility, 50.0)
    self.assertEqual(CONFIG.param_filter_gamma_theta, 0.01)
    self.assertEqual(CONFIG.param_filter_gamma_f, 0.01)