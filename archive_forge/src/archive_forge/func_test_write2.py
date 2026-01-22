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
def test_write2(self):
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.x = Var(model.A, bounds=(-1, 1))

    def obj_rule(model):
        return sum_product(model.x)
    model.obj = Objective(rule=obj_rule)

    def c_rule(model):
        return (1, model.x[1] + model.x[2], 2)
    model.c = Constraint(rule=c_rule)
    model.write()