import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_indicator_var_references(self):
    m = models.makeTwoTermDisj()
    m.simple = Constraint(expr=m.x <= m.a + 1)
    m.basic_step = apply_basic_step([m.disjunction, m.simple])
    refs = [v for v in m.basic_step.component_data_objects(BooleanVar, sort=SortComponents.deterministic)]
    self.assertEqual(len(refs), 2)
    self.assertIs(refs[0][None], m.d[0].indicator_var)
    self.assertIs(refs[1][None], m.d[1].indicator_var)