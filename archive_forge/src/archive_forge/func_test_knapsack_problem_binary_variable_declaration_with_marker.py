import os
import random
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_knapsack_problem_binary_variable_declaration_with_marker(self):
    elements_size = [30, 24, 11, 35, 29, 8, 31, 18]
    elements_weight = [3, 2, 2, 4, 5, 4, 3, 1]
    capacity = 60
    model = ConcreteModel('knapsack problem')
    var_names = [f'{i + 1}' for i in range(len(elements_size))]
    model.x = Var(var_names, within=Binary)
    model.obj = Objective(expr=sum((model.x[var_names[i]] * elements_weight[i] for i in range(len(elements_size)))), sense=minimize, name='obj')
    model.const1 = Constraint(expr=sum((model.x[var_names[i]] * elements_size[i] for i in range(len(elements_size)))) >= capacity, name='const')
    self._check_baseline(model, int_marker=True)