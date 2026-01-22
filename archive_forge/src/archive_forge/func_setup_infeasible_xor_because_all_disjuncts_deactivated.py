import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def setup_infeasible_xor_because_all_disjuncts_deactivated(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.y = Var(bounds=(0, 7))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])
    m.disjunction_disjuncts[0].nestedDisjunction = Disjunction(expr=[m.y == 6, m.y <= 1])
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[1].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=m.disjunction.disjuncts[0].nestedDisjunction)
    xor = m.disjunction_disjuncts[0].nestedDisjunction.algebraic_constraint
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(value(xor.lower), 1)
    self.assertEqual(value(xor.upper), 1)
    repn = generate_standard_repn(xor.body)
    for v in repn.linear_vars:
        self.assertTrue(v.is_fixed())
        self.assertEqual(value(v), 0)
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    return m