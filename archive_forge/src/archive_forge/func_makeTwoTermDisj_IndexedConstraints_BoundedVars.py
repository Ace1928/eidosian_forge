from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj_IndexedConstraints_BoundedVars():
    """Single two-term disjunction with IndexedConstraints on both disjuncts."""
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.lbs = Param(m.s, initialize={1: 2, 2: 4})
    m.ubs = Param(m.s, initialize={1: 7, 2: 6})

    def bounds_rule(m, s):
        return (m.lbs[s], m.ubs[s])
    m.a = Var(m.s, bounds=bounds_rule)

    def d_rule(disjunct, flag):
        m = disjunct.model()

        def true_rule(d, s):
            return m.a[s] == 0

        def false_rule(d, s):
            return m.a[s] >= 5
        if flag:
            disjunct.c = Constraint(m.s, rule=true_rule)
        else:
            disjunct.c = Constraint(m.s, rule=false_rule)
    m.disjunct = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.disjunct[0], m.disjunct[1]])
    return m