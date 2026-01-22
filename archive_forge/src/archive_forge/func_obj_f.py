import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
def obj_f(m):
    return sum((m.param_cx[e] * m.x[e] for e in m.E)) + sum((m.param_cy[a] * m.y[a] for a in m.A))