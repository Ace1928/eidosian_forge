from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeDisjunctWithExpression():
    """Two-term SimpleDisjunction where one of the disjuncts contains an
    Expression. This is used to make sure that we correctly handle types we
    hit in disjunct.component_objects(active=True)"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.e = Expression(expr=m.x ** 2)
    m.d1.c = Constraint(rule=lambda _: m.x == 1)
    m.d2 = Disjunct()
    m.disj = Disjunction(expr=[m.d1, m.d2])
    return m