from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disjunction_rule(m, s):
    return [m.disjunct[s, flag] for flag in [0, 1]]