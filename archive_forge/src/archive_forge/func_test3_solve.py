from itertools import zip_longest
import json
import os
import sys
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test3_solve(self):
    if not 'glpk' in solvers:
        self.skipTest('glpk solver is not available')
    self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
    opt = pyomo.opt.SolverFactory('glpk')
    _test = TempfileManager.create_tempfile(suffix='test3.out')
    results = opt.solve(self.model, keepfiles=False)
    results.write(filename=_test, format='json')
    with open(_test, 'r') as out, open(join(currdir, 'test3.baseline.out'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=1e-06, allow_second_superset=True)