import os
import pyomo.common.unittest as unittest
from io import StringIO
import logging
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface
@unittest.skipIf(not opt_k_aug.available(), 'k_aug is not available')
def test_clear_dir_k_aug(self):
    m = simple_model_1()
    sens = SensitivityInterface(m, clone_model=False)
    k_aug = K_augInterface()
    opt_ipopt.solve(m, tee=True)
    m.ptb = pyo.Param(mutable=True, initialize=1.5)
    cwd = os.getcwd()
    dir_contents = os.listdir(cwd)
    sens_param = [m.p]
    sens.setup_sensitivity(sens_param)
    k_aug.k_aug(m, tee=True)
    self.assertEqual(cwd, os.getcwd())
    self.assertEqual(dir_contents, os.listdir(cwd))
    self.assertFalse(os.path.exists('dsdp_in_.in'))
    self.assertFalse(os.path.exists('conorder.txt'))
    self.assertFalse(os.path.exists('timings_k_aug_dsdp.txt'))
    self.assertIsInstance(k_aug.data['dsdp_in_.in'], str)
    self.assertIsInstance(k_aug.data['conorder.txt'], str)
    self.assertIsInstance(k_aug.data['timings_k_aug_dsdp.txt'], str)