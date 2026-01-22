from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def twoSegments_SawayaGrossmann():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3))
    m.disj1 = Disjunct()
    m.disj1.c = Constraint(expr=inequality(0, m.x, 1))
    m.disj2 = Disjunct()
    m.disj2.c = Constraint(expr=inequality(2, m.x, 3))
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
    m.obj = Objective(expr=m.x - m.disj2.indicator_var)
    return m