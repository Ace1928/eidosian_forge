from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeDisjunctionOfDisjunctDatas():
    """Two SimpleDisjunctions, where each are disjunctions of DisjunctDatas.
    This adds nothing to makeTwoSimpleDisjunctions but exists for convenience
    because it has the same mathematical meaning as
    makeAnyIndexedDisjunctionOfDisjunctDatas
    """
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))
    m.obj = Objective(expr=m.x)
    m.idx = Set(initialize=[1, 2])
    m.firstTerm = Disjunct(m.idx)
    m.firstTerm[1].cons = Constraint(expr=m.x == 0)
    m.firstTerm[2].cons = Constraint(expr=m.x == 2)
    m.secondTerm = Disjunct(m.idx)
    m.secondTerm[1].cons = Constraint(expr=m.x >= 2)
    m.secondTerm[2].cons = Constraint(expr=m.x >= 3)
    m.disjunction = Disjunction(expr=[m.firstTerm[1], m.secondTerm[1]])
    m.disjunction2 = Disjunction(expr=[m.firstTerm[2], m.secondTerm[2]])
    return m