import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def check_after_improper_basic_step(self, m):
    for disj in m.basic_step.disjuncts.values():
        self.assertEqual(len(disj.improper_constraints), 1)
        cons = disj.improper_constraints[1]
        self.check_constraint_body(m, cons, -1)