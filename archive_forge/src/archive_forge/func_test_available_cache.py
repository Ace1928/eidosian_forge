import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
def test_available_cache(self):
    opt = ipopt.Ipopt()
    opt.available()
    self.assertTrue(opt._available_cache[1])
    self.assertIsNotNone(opt._available_cache[0])
    config = ipopt.IpoptConfig()
    config.executable = Executable('/a/bogus/path')
    opt.available(config=config)
    self.assertFalse(opt._available_cache[1])
    self.assertIsNone(opt._available_cache[0])