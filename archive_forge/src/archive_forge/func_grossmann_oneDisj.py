from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def grossmann_oneDisj():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 20))
    m.y = Var(bounds=(0, 20))
    m.disjunct1 = Disjunct()
    m.disjunct1.constraintx = Constraint(expr=inequality(0, m.x, 2))
    m.disjunct1.constrainty = Constraint(expr=inequality(7, m.y, 10))
    m.disjunct2 = Disjunct()
    m.disjunct2.constraintx = Constraint(expr=inequality(8, m.x, 10))
    m.disjunct2.constrainty = Constraint(expr=inequality(0, m.y, 3))
    m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])
    m.objective = Objective(expr=m.x + 2 * m.y, sense=maximize)
    return m