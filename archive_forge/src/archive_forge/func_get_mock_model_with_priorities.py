import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def get_mock_model_with_priorities(self):
    m = pmo.block()
    m.x = pmo.variable(domain=Integers)
    m.s = range(10)
    m.y = pmo.variable_list((pmo.variable(domain=Integers) for _ in m.s))
    m.o = pmo.objective(expr=m.x + sum(m.y), sense=minimize)
    m.c = pmo.constraint(expr=m.x >= 1)
    m.c2 = pmo.constraint(expr=quicksum((m.y[i] for i in m.s)) >= 10)
    m.priority = pmo.suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    m.direction = pmo.suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    m.priority[m.x] = 1
    m.priority[m.y] = 2
    m.direction[m.y] = BranchDirection.down
    m.direction[m.y[-1]] = BranchDirection.up
    return m