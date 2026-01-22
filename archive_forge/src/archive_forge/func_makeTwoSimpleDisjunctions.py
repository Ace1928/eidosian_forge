from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoSimpleDisjunctions():
    """Two SimpleDisjunctions on the same model."""
    m = ConcreteModel()
    m.a = Var(bounds=(-10, 50))

    def d1_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a == 0)
        else:
            disjunct.c = Constraint(expr=m.a >= 5)
    m.disjunct1 = Disjunct([0, 1], rule=d1_rule)

    def d2_rule(disjunct, flag):
        if not flag:
            disjunct.c = Constraint(expr=m.a >= 30)
        else:
            disjunct.c = Constraint(expr=m.a == 100)
    m.disjunct2 = Disjunct([0, 1], rule=d2_rule)
    m.disjunction1 = Disjunction(expr=[m.disjunct1[0], m.disjunct1[1]])
    m.disjunction2 = Disjunction(expr=[m.disjunct2[0], m.disjunct2[1]])
    return m