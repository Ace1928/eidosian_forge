from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
@unittest.skipUnless(gurobi_available, 'Gurobi is not available')
def test_var_bounds_substituted_for_missing_bound_constraints(self):
    m = self.make_model()
    self.add_fourth_disjunct(m)
    mbm = TransformationFactory('gdp.mbigm')
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.gdp.mbigm'):
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=True)
    warnings = out.getvalue()
    self.assertIn('Unused arguments in the bigM map! These arguments were not used by the transformation:', warnings)
    for cons, disj in [(m.d1.x1_bounds, m.d2), (m.d1.x2_bounds, m.d2), (m.d1.x1_bounds, m.d3), (m.d1.x2_bounds, m.d3), (m.d2.x1_bounds, m.d1), (m.d2.x2_bounds, m.d1), (m.d2.x1_bounds, m.d3), (m.d2.x2_bounds, m.d3), (m.d3.x1_bounds, m.d1), (m.d3.x2_bounds, m.d1), (m.d3.x1_bounds, m.d2), (m.d3.x2_bounds, m.d2)]:
        self.assertIn('(%s, %s)' % (cons.name, disj.name), warnings)
    cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
    self.assertEqual(len(cons), 2)
    sameish = mbm.get_transformed_constraints(m.d4.x1_ub)
    self.assertEqual(len(sameish), 1)
    self.assertIs(sameish[0], cons[1])
    self.check_pretty_bound_constraints(cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10, m.d4: 8}, lb=False)
    self.check_pretty_bound_constraints(cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2, m.d4: -10}, lb=True)
    cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
    self.assertEqual(len(cons), 2)
    sameish = mbm.get_transformed_constraints(m.d4.x2_lb)
    self.assertEqual(len(sameish), 1)
    self.assertIs(sameish[0], cons[0])
    self.check_pretty_bound_constraints(cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1, m.d4: 20}, lb=False)
    self.check_pretty_bound_constraints(cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55, m.d4: -5}, lb=True)