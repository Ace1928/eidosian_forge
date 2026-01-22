import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def check_second_disjunction_aux_vars(self, aux_vars1, aux_vars2):
    self.assertEqual(len(aux_vars1), 2)
    self.assertEqual(aux_vars1[0].lb, -1)
    self.assertEqual(aux_vars1[0].ub, 24)
    self.assertEqual(aux_vars1[1].lb, 0)
    self.assertEqual(aux_vars1[1].ub, 36)
    self.assertEqual(len(aux_vars2), 2)
    self.assertEqual(aux_vars2[0].lb, -4)
    self.assertEqual(aux_vars2[0].ub, 12)
    self.assertEqual(aux_vars2[1].lb, -9)
    self.assertEqual(aux_vars2[1].ub, 16)