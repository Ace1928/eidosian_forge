import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
def test_custom_instantiation(self):
    config = ipopt.IpoptConfig(description='A description')
    config.tee = True
    self.assertTrue(config.tee)
    self.assertEqual(config._description, 'A description')
    self.assertIsNone(config.time_limit)
    self.assertIsNotNone(str(config.executable))
    self.assertIn('ipopt', str(config.executable))
    config.executable = Executable('/bogus/path')
    self.assertIsNone(config.executable.executable)
    self.assertFalse(config.executable.available())