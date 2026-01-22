from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def d1_rule(disjunct, flag):
    m = disjunct.model()
    if flag:
        disjunct.c = Constraint(expr=m.a == 0)
    else:
        disjunct.c = Constraint(expr=m.a >= 5)