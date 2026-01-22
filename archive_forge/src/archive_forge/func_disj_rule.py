from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disj_rule(m, t):
    return [[m.x[t] == 1], [m.x[t] == 2]]