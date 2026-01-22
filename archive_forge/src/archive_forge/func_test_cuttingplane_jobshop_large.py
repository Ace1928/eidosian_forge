import os
import sys
from os.path import abspath, dirname, normpath, join
from pyomo.common.fileutils import import_file
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
from pyomo.environ import SolverFactory, TransformationFactory
@unittest.skip('cutting plane LP file tests are too fragile')
@unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
def test_cuttingplane_jobshop_large(self):
    self.problem = 'test_cuttingplane_jobshop_large'
    self.pyomo('jobshop.dat', preprocess='cuttingplane')
    self.check('jobshop_large', 'cuttingplane')