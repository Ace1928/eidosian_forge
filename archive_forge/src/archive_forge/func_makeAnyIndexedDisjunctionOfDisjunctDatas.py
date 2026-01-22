from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeAnyIndexedDisjunctionOfDisjunctDatas():
    """An IndexedDisjunction indexed by Any, with two two-term DisjunctionDatas
    build from DisjunctDatas. Identical mathematically to
    makeDisjunctionOfDisjunctDatas.

    Used to test that the right things happen for a case where soemone
    implements an algorithm which iteratively generates disjuncts and
    retransforms"""
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
    m.disjunction = Disjunction(Any)
    m.disjunction[1] = [m.firstTerm[1], m.secondTerm[1]]
    m.disjunction[2] = [m.firstTerm[2], m.secondTerm[2]]
    return m