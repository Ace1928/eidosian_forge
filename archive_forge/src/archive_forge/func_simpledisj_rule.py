from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def simpledisj_rule(disjunct):
    m = disjunct.model()

    @disjunct.Disjunct()
    def innerdisjunct0(disjunct):
        disjunct.c = Constraint(expr=m.x <= 2)

    @disjunct.Disjunct()
    def innerdisjunct1(disjunct):
        disjunct.c = Constraint(expr=m.x >= 4)
    disjunct.innerdisjunction = Disjunction(expr=[disjunct.innerdisjunct0, disjunct.innerdisjunct1])