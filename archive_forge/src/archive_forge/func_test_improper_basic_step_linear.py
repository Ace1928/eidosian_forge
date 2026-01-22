import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_improper_basic_step_linear(self):
    model_builder = import_file(join(exdir, 'two_rxn_lee', 'two_rxn_model.py'))
    m = model_builder.build_model(use_mccormick=True)
    m.basic_step = apply_basic_step([m.reactor_choice, m.max_demand, m.mccormick_1, m.mccormick_2])
    for disj in m.basic_step.disjuncts.values():
        self.assertIs(disj.improper_constraints[1].body, m.P)
        self.assertEqual(disj.improper_constraints[1].lower, None)
        self.assertEqual(disj.improper_constraints[1].upper, 2)
        self.assertEqual(disj.improper_constraints[2].body.polynomial_degree(), 1)
        self.assertEqual(disj.improper_constraints[2].lower, None)
        self.assertEqual(disj.improper_constraints[2].upper, 0)
        self.assertEqual(disj.improper_constraints[3].body.polynomial_degree(), 1)
        self.assertEqual(disj.improper_constraints[3].lower, None)
        self.assertEqual(disj.improper_constraints[3].upper, 0)
        self.assertEqual(len(disj.improper_constraints), 3)
    self.assertFalse(m.max_demand.active)
    self.assertFalse(m.mccormick_1.active)
    self.assertFalse(m.mccormick_2.active)