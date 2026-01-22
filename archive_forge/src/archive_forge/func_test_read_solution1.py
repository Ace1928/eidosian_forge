import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
@unittest.skipIf(not yaml_available, "Cannot import 'yaml'")
def test_read_solution1(self):
    """Read a SolverResults Object"""
    self.results = pyomo.opt.SolverResults()
    self.results.read(filename=join(currdir, 'test4_sol.txt'))
    self.results.write(filename=join(currdir, 'read_solution1.out'))
    if not os.path.exists(join(currdir, 'read_solution1.out')):
        self.fail('test_read_solution1 - failed to write read_solution1.out')
    with open(join(currdir, 'read_solution1.out'), 'r') as out, open(join(currdir, 'test4_sol.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True)