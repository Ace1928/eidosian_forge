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
def test_write3(self):
    model = ConcreteModel()
    model.J = RangeSet(1, 4)
    model.w = Param(model.J, default=4)
    model.x = Var(model.J, initialize=3)

    def obj_rule(instance):
        return sum_product(instance.w, instance.x)
    model.obj = Objective(rule=obj_rule)
    self.assertEqual(value(model.obj), 48)