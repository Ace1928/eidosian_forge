import os
import random
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_integer_variable_declaration_with_marker(self):
    model = ConcreteModel('Example-mix-integer-linear-problem')
    model.x1 = Var(within=NonNegativeIntegers)
    model.x2 = Var(within=NonNegativeReals)
    model.obj = Objective(expr=3 * model.x1 + 2 * model.x2, sense=minimize)
    model.const1 = Constraint(expr=4 * model.x1 + 3 * model.x2 >= 10)
    model.const2 = Constraint(expr=model.x1 + 2 * model.x2 <= 7)
    self._check_baseline(model, int_marker=True)