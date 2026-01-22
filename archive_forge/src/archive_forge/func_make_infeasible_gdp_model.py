from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def make_infeasible_gdp_model():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.d = Disjunction(expr=[[m.x ** 2 >= 3, m.x >= 3], [m.x ** 2 <= -1, m.x <= -1]])
    m.o = Objective(expr=m.x)
    return m