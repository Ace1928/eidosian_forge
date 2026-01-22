import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def modify_two_term_disjunction(self, m):
    m.a.setlb(0)
    m.y = Var(domain=Integers, bounds=(2, 4))
    m.d[1].c3 = Constraint(expr=m.x <= 6)
    m.d[0].c2 = Constraint(expr=m.y + m.a - 5 <= 2)
    m.obj = Objective(expr=-m.x - m.y)