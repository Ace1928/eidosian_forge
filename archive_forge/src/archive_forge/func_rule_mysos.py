import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
def rule_mysos(m):
    return ([m.y[a] for a in m.y], [m.mysosweights[a] for a in m.y])