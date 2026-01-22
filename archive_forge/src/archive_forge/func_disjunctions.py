from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@m.Disjunction([0, 1])
def disjunctions(m, i):
    if i == 0:
        return Disjunction.Skip
    return [m.disjuncts[i], m.disjuncts[0]]