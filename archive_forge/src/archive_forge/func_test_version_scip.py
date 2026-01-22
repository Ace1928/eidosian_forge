import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
def test_version_scip(self):
    self.assertTrue(self.scip.version() is not None)
    self.assertTrue(type(self.scip.version()) is tuple)
    self.assertEqual(len(self.scip.version()), 4)