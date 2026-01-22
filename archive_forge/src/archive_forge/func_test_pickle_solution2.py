import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_pickle_solution2(self):
    """Read a SolverResults Object"""
    self.results = pyomo.opt.SolverResults()
    self.results.read(filename=join(currdir, 'test4_sol.jsn'), format='json')
    str = pickle.dumps(self.results)
    res = pickle.loads(str)
    self.results.write(filename=join(currdir, 'read_solution2.out'), format='json')
    if not os.path.exists(join(currdir, 'read_solution2.out')):
        self.fail('test_read_solution2 - failed to write read_solution2.out')
    with open(join(currdir, 'read_solution2.out'), 'r') as out, open(join(currdir, 'test4_sol.jsn'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), allow_second_superset=True)