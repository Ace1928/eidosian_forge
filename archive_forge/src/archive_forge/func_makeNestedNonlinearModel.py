from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNestedNonlinearModel():
    """This is actually a disjunction between two points, but it's written
    as a nested disjunction over four circles!"""
    m = ConcreteModel()
    m.x = Var(bounds=(-10, 10))
    m.y = Var(bounds=(-10, 10))
    m.d1 = Disjunct()
    m.d1.lower_circle = Constraint(expr=m.x ** 2 + m.y ** 2 <= 1)
    m.disj = Disjunction(expr=[[m.x == 10], [(sqrt(2) - m.x) ** 2 + (sqrt(2) - m.y) ** 2 <= 1]])
    m.d2 = Disjunct()
    m.d2.upper_circle = Constraint(expr=(3 - m.x) ** 2 + (3 - m.y) ** 2 <= 1)
    m.d2.inner = Disjunction(expr=[[m.y == 10], [(sqrt(2) - m.x) ** 2 + (sqrt(2) - m.y) ** 2 <= 1]])
    m.outer = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=m.x + m.y)
    return m