from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_lower_bound_unsat(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 5))
    m.c = Constraint(expr=-0.01 == m.x)
    m.o = Objective(expr=m.x)
    self.assertFalse(satisfiable(m))