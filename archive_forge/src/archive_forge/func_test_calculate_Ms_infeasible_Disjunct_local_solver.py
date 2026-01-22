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
@unittest.skipUnless(SolverFactory('ipopt').available(exception_flag=False), 'Ipopt is not available')
def test_calculate_Ms_infeasible_Disjunct_local_solver(self):
    m = self.make_infeasible_disjunct_model()
    with self.assertRaisesRegex(GDP_Error, "Unsuccessful solve to calculate M value to relax constraint 'disjunction_disjuncts\\[1\\].constraint\\[1\\]' on Disjunct 'disjunction_disjuncts\\[1\\]' when Disjunct 'disjunction_disjuncts\\[0\\]' is selected."):
        TransformationFactory('gdp.mbigm').apply_to(m, solver=SolverFactory('ipopt'), reduce_bound_constraints=False)