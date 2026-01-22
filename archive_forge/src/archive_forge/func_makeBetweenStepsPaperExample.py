from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeBetweenStepsPaperExample():
    """Original example model, implicit disjunction"""
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.x = Var(m.I, bounds=(-2, 6))
    m.disjunction = Disjunction(expr=[[sum((m.x[i] ** 2 for i in m.I)) <= 1], [sum(((3 - m.x[i]) ** 2 for i in m.I)) <= 1]])
    m.obj = Objective(expr=m.x[2] - m.x[1], sense=maximize)
    return m