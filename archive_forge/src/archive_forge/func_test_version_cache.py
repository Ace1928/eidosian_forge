import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
def test_version_cache(self):
    opt = ipopt.Ipopt()
    opt.version()
    self.assertIsNotNone(opt._version_cache[0])
    self.assertIsNotNone(opt._version_cache[1])
    config = ipopt.IpoptConfig()
    config.executable = Executable('/a/bogus/path')
    opt.version(config=config)
    self.assertIsNone(opt._version_cache[0])
    self.assertIsNone(opt._version_cache[1])