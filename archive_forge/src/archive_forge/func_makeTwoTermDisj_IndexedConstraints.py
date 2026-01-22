from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj_IndexedConstraints():
    """Single two-term disjunction with IndexedConstraints on both disjuncts.
    Does not bound the variables, so cannot be transformed by hull at all and
    requires specifying m values in bigm.
    """
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s)
    m.b = Block()

    def disj1_rule(disjunct):
        m = disjunct.model()

        def c_rule(d, s):
            return m.a[s] == 0
        disjunct.c = Constraint(m.s, rule=c_rule)
    m.b.simpledisj1 = Disjunct(rule=disj1_rule)

    def disj2_rule(disjunct):
        m = disjunct.model()

        def c_rule(d, s):
            return m.a[s] <= 3
        disjunct.c = Constraint(m.s, rule=c_rule)
    m.b.simpledisj2 = Disjunct(rule=disj2_rule)
    m.b.disjunction = Disjunction(expr=[m.b.simpledisj1, m.b.simpledisj2])
    return m