import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_get_solution(self):
    """Get a solution from a SolverResults object"""
    tmp = self.results.solution[0]
    self.assertEqual(tmp, self.soln)