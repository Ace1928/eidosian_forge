import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_improper_basic_step(self):
    model_builder = import_file(join(exdir, 'two_rxn_lee', 'two_rxn_model.py'))
    m = model_builder.build_model()
    m.basic_step = apply_basic_step([m.reactor_choice, m.max_demand])
    for disj in m.basic_step.disjuncts.values():
        self.assertEqual(disj.improper_constraints[1].body.polynomial_degree(), 2)
        self.assertEqual(disj.improper_constraints[1].lower, None)
        self.assertEqual(disj.improper_constraints[1].upper, 2)
        self.assertEqual(len(disj.improper_constraints), 1)
    self.assertFalse(m.max_demand.active)