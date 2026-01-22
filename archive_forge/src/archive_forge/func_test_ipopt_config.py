import pyomo.environ as pyo
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver.ipopt import IpoptConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.common import unittest
def test_ipopt_config(self):
    config = IpoptConfig()
    self.assertTrue(config.load_solutions)
    self.assertIsInstance(config.solver_options, ConfigDict)
    self.assertIsInstance(config.executable, ExecutableData)
    solver = SolverFactory('ipopt_v2', executable='/path/to/exe')
    self.assertFalse(solver.config.tee)
    self.assertTrue(solver.config.executable.startswith('/path'))