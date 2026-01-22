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
def check_untightened_bounds_constraint(self, cons, var, parent_disj, disjunction, Ms, lower=None, upper=None):
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 3)
    self.assertIsNone(cons.lower)
    self.assertEqual(value(cons.upper), 0)
    if lower is not None:
        self.assertEqual(repn.constant, lower)
        check_linear_coef(self, repn, var, -1)
        for disj in disjunction.disjuncts:
            if disj is not parent_disj:
                check_linear_coef(self, repn, disj.binary_indicator_var, Ms[disj] - lower)
    if upper is not None:
        self.assertEqual(repn.constant, -upper)
        check_linear_coef(self, repn, var, 1)
        for disj in disjunction.disjuncts:
            if disj is not parent_disj:
                check_linear_coef(self, repn, disj.binary_indicator_var, -Ms[disj] + upper)