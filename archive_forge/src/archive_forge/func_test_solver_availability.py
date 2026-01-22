import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
@unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
def test_solver_availability(self):
    self.instance = base.SolverBase()
    self.instance.Availability._value_ = 1
    self.assertTrue(self.instance.Availability.__bool__(self.instance.Availability))
    self.instance.Availability._value_ = -1
    self.assertFalse(self.instance.Availability.__bool__(self.instance.Availability))