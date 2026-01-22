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
def check_transformation_block_structure(self, m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub):
    b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m, m.disjunction)
    self.assertEqual(len(disj1.component_map(Var)), 3)
    self.assertEqual(len(disj2.component_map(Var)), 3)
    aux_vars1 = disj1.component('disjunction_disjuncts[0].constraint[1]_aux_vars')
    aux_vars2 = disj2.component('disjunction_disjuncts[1].constraint[1]_aux_vars')
    self.check_aux_var_bounds(aux_vars1, aux_vars2, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub)
    return (b, disj1, disj2, aux_vars1, aux_vars2)