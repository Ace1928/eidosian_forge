from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_disjunction_sat1(self):
    m = ConcreteModel()
    m.x1 = Var(bounds=(0, 8))
    m.x2 = Var(bounds=(0, 8))
    m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
    m.y1 = Disjunct()
    m.y2 = Disjunct()
    m.y1.c1 = Constraint(expr=m.x1 >= 9)
    m.y1.c2 = Constraint(expr=m.x2 >= 2)
    m.y2.c1 = Constraint(expr=m.x1 >= 3)
    m.y2.c2 = Constraint(expr=m.x2 >= 3)
    m.djn = Disjunction(expr=[m.y1, m.y2])
    self.assertTrue(satisfiable(m))