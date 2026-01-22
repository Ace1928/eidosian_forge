import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
@unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
def test_config_kwds(self):
    self.instance = base.SolverBase(tee=True)
    self.assertTrue(self.instance.config.tee)