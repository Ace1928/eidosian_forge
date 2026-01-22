import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_improper_basic_step_indexedConstraint(self):
    m = models.makeTwoTermDisj()

    @m.Constraint([1, 2])
    def indexed(m, i):
        return m.x <= m.a + i
    m.basic_step = apply_basic_step([m.disjunction, m.indexed])
    for disj in m.basic_step.disjuncts.values():
        self.assertEqual(len(disj.improper_constraints), 2)
        cons = disj.improper_constraints[1]
        self.check_constraint_body(m, cons, -1)
        cons = disj.improper_constraints[2]
        self.check_constraint_body(m, cons, -2)