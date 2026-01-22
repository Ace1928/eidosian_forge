import os
import sys
from os.path import abspath, dirname, normpath, join
from pyomo.common.fileutils import import_file
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
from pyomo.environ import SolverFactory, TransformationFactory
def getObjective(self, fname):
    FILE = open(fname)
    data = yaml.load(FILE, **yaml_load_args)
    FILE.close()
    solutions = data.get('Solution', [])
    ans = []
    for x in solutions:
        ans.append(x.get('Objective', {}))
    return ans