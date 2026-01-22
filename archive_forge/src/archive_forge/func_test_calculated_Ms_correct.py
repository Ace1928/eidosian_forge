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
def test_calculated_Ms_correct(self):
    m = self.make_model()
    mbm = TransformationFactory('gdp.mbigm')
    mbm.apply_to(m, reduce_bound_constraints=False)
    self.check_all_untightened_bounds_constraints(m, mbm)
    self.check_linear_func_constraints(m, mbm)
    self.assertStructuredAlmostEqual(mbm.get_all_M_values(m), self.get_Ms(m))