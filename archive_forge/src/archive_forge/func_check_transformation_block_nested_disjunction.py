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
def check_transformation_block_nested_disjunction(self, m, disj2, x, disjunction_block=None):
    if disjunction_block is None:
        block_prefix = ''
        disjunction_parent = m
    else:
        block_prefix = disjunction_block + '.'
        disjunction_parent = m.component(disjunction_block)
    inner_b, inner_disj1, inner_disj2 = self.check_transformation_block_disjuncts_and_constraints(disj2, disjunction_parent.disj2.disjunction, '%sdisj2.disjunction' % block_prefix)
    self.assertEqual(len(inner_disj1.component_map(Var)), 3)
    self.assertEqual(len(inner_disj2.component_map(Var)), 3)
    aux_vars1 = inner_disj1.component('%sdisj2.disjunction_disjuncts[0].constraint[1]_aux_vars' % block_prefix)
    aux_vars2 = inner_disj2.component('%sdisj2.disjunction_disjuncts[1].constraint[1]_aux_vars' % block_prefix)
    self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
    c = inner_disj1.component('%sdisj2.disjunction_disjuncts[0].constraint[1]' % block_prefix)
    self.assertEqual(len(c), 1)
    c1 = c[0]
    self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
    c = inner_disj2.component('%sdisj2.disjunction_disjuncts[1].constraint[1]' % block_prefix)
    self.assertEqual(len(c), 1)
    c2 = c[0]
    self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])
    c = inner_b.component('%sdisj2.disjunction_disjuncts[0].constraint[1]_split_constraints' % block_prefix)
    self.assertEqual(len(c), 2)
    c1 = c[0]
    self.check_global_constraint_disj1(c1, aux_vars1[0], x[1], x[2])
    c2 = c[1]
    self.check_global_constraint_disj1(c2, aux_vars1[1], x[3], x[4])
    c = inner_b.component('%sdisj2.disjunction_disjuncts[1].constraint[1]_split_constraints' % block_prefix)
    self.assertEqual(len(c), 2)
    c1 = c[0]
    self.check_global_constraint_disj2(c1, aux_vars2[0], x[1], x[2])
    c2 = c[1]
    self.check_global_constraint_disj2(c2, aux_vars2[1], x[3], x[4])