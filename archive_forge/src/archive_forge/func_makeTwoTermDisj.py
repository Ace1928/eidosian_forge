from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj():
    """Single two-term disjunction which has all of ==, <=, and >= constraints"""
    m = ConcreteModel()
    m.a = Var(bounds=(2, 7))
    m.x = Var(bounds=(4, 9))

    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=m.a == 0)
            disjunct.c2 = Constraint(expr=m.x <= 7)
        else:
            disjunct.c = Constraint(expr=m.a >= 5)
    m.d = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[0], m.d[1]])
    return m