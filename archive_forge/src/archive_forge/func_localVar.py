from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def localVar():
    """Two-term disjunction which declares a local variable y on one of the
    disjuncts, which is used in the objective function as well.

    Used to test that we will treat y as global in the transformations,
    despite where it is declared.
    """
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3))
    m.disj1 = Disjunct()
    m.disj1.cons = Constraint(expr=m.x >= 1)
    m.disj2 = Disjunct()
    m.disj2.y = Var(bounds=(1, 3))
    m.disj2.cons = Constraint(expr=m.x + m.disj2.y == 3)
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
    m.objective = Objective(expr=m.x + m.disj2.y)
    return m