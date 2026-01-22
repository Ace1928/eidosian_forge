from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeThreeTermDisjunctionWithOneVarInOneDisjunct():
    """This is to make sure hull doesn't create more disaggregated variables
    than it needs to: Here, x only appears in the first Disjunct, so we only
    need two copies: one as usual for that disjunct and then one other that is
    free if either of the second two Disjuncts is active and 0 otherwise.
    """
    m = ConcreteModel()
    m.x = Var(bounds=(-2, 8))
    m.y = Var(bounds=(3, 4))
    m.d1 = Disjunct()
    m.d1.c1 = Constraint(expr=m.x <= 3)
    m.d1.c2 = Constraint(expr=m.y >= 3.5)
    m.d2 = Disjunct()
    m.d2.c1 = Constraint(expr=m.y >= 3.7)
    m.d3 = Disjunct()
    m.d3.c1 = Constraint(expr=m.y >= 3.9)
    m.disjunction = Disjunction(expr=[m.d1, m.d2, m.d3])
    return m