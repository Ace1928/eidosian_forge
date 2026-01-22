import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_dlog_bad_length(self):
    model = ConcreteModel()
    model.range = Var()
    model.x = Var(bounds=(-1, 1))
    args = (model.range, model.x)
    keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'pw_repn': 'DLOG', 'f_rule': lambda model, x: x ** 2}
    model.con = Piecewise(*args, **keywords)
    try:
        keywords['pw_pts'] = [-1, 0, 0.5, 1]
        model.con3 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized with DLOG an pw_pts list with length not equal to (2^n)+1.')