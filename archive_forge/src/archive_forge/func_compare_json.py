import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
def compare_json(self, file1, file2):
    with open(file1, 'r') as out, open(file2, 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=1e-07, allow_second_superset=True)