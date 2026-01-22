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
def test_calculate_Ms_infeasible_Disjunct(self):
    m = self.make_infeasible_disjunct_model()
    out = StringIO()
    mbm = TransformationFactory('gdp.mbigm')
    with LoggingIntercept(out, 'pyomo.gdp.mbigm', logging.DEBUG):
        mbm.apply_to(m, reduce_bound_constraints=False)
    self.assertIn("Disjunct 'disjunction_disjuncts[0]' is infeasible, deactivating", out.getvalue().strip())
    self.assertFalse(m.disjunction.disjuncts[0].active)
    self.assertTrue(m.disjunction.disjuncts[0].indicator_var.fixed)
    self.assertFalse(value(m.disjunction.disjuncts[0].indicator_var))
    cons = mbm.get_transformed_constraints(m.disjunction.disjuncts[1].constraint[1])
    self.assertEqual(len(cons), 1)
    assertExpressionsEqual(self, cons[0].expr, 21 + m.x - m.y <= 0 * m.disjunction.disjuncts[0].binary_indicator_var + 12.0 * m.disjunction.disjuncts[2].binary_indicator_var)
    cons = mbm.get_transformed_constraints(m.disjunction.disjuncts[2].constraint[1])
    self.assertEqual(len(cons), 2)
    print(cons[0].expr)
    print(cons[1].expr)
    assertExpressionsEqual(self, cons[0].expr, 0.0 * m.disjunction_disjuncts[0].binary_indicator_var - 12.0 * m.disjunction_disjuncts[1].binary_indicator_var <= m.x - (m.y - 9))
    assertExpressionsEqual(self, cons[1].expr, m.x - (m.y - 9) <= 0.0 * m.disjunction_disjuncts[0].binary_indicator_var - 12.0 * m.disjunction_disjuncts[1].binary_indicator_var)