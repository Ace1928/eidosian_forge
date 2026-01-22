import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
def constr_y_lb(m):
    return m.x[1] + m.x[2] + m.y[1] + m.y[2] + m.y[5] + m.y[6] >= 0.25