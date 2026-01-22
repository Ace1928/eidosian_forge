from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disjunction1_rule(m, s):
    return [m.disjunct1[s, flag] for flag in [0, 1]]