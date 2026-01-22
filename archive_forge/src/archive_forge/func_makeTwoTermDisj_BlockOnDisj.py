from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj_BlockOnDisj():
    """SimpleDisjunction where one of the Disjuncts contains three different
    blocks: two simple and one indexed"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1000))
    m.y = Var(bounds=(0, 800))

    def disj_rule(d, flag):
        m = d.model()
        if flag:
            d.b = Block()
            d.b.c = Constraint(expr=m.x == 0)
            d.add_component('b.c', Constraint(expr=m.y >= 9))
            d.b.anotherblock = Block()
            d.b.anotherblock.c = Constraint(expr=m.y >= 11)
            d.bb = Block([1])
            d.bb[1].c = Constraint(expr=m.x == 0)
        else:
            d.c = Constraint(expr=m.x >= 80)
    m.evil = Disjunct([0, 1], rule=disj_rule)
    m.disjunction = Disjunction(expr=[m.evil[0], m.evil[1]])
    return m