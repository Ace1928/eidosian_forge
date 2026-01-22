import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
@unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
def test_equality_constraints_on_disjuncts_with_fme(self):
    m = models.oneVarDisj_2pts()
    m.obj.expr = m.x + m.disj1.indicator_var
    m.obj.sense = maximize
    TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None, verbose=True, solver='gurobi', solver_options={'FeasibilityTol': 1e-08}, cuts_name='cuts', bigM=5)
    self.assertEqual(len(m.cuts), 1)
    cut = m.cuts[0]
    self.assertEqual(cut.lower, 0)
    self.assertIsNone(cut.upper)
    repn = generate_standard_repn(cut.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertIs(repn.linear_vars[0], m.disj1.binary_indicator_var)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertIs(repn.linear_vars[1], m.x)
    self.assertEqual(repn.linear_coefs[1], -1)