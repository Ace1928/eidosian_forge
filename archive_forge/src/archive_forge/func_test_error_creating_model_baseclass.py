import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def test_error_creating_model_baseclass(self):
    with self.assertRaisesRegex(TypeError, "Directly creating the 'Model' class is not allowed.  Please use the AbstractModel or ConcreteModel class instead."):
        m = Model()