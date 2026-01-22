from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def simpdisj_rule(disjunct):
    m = disjunct.model()
    disjunct.c = Constraint(expr=m.a >= 3)