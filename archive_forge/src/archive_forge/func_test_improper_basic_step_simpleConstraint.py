import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_improper_basic_step_simpleConstraint(self):
    m = models.makeTwoTermDisj()
    m.simple = Constraint(expr=m.x <= m.a + 1)
    m.basic_step = apply_basic_step([m.disjunction, m.simple])
    self.check_after_improper_basic_step(m)
    self.assertFalse(m.simple.active)
    self.assertFalse(m.disjunction.active)