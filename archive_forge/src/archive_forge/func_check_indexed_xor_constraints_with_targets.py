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
def check_indexed_xor_constraints_with_targets(self, transformation):
    m = models.makeTwoTermIndexedDisjunction_BoundedVars()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disjunction[1], m.disjunction[3]])
    xorC = m.disjunction[1].algebraic_constraint.parent_component()
    self.assertIsInstance(xorC, Constraint)
    self.assertEqual(len(xorC), 2)
    for i in [1, 3]:
        self.assertEqual(xorC[i].lower, 1)
        self.assertEqual(xorC[i].upper, 1)
        repn = generate_standard_repn(xorC[i].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, m.disjunct[i, 0].indicator_var, 1)
        check_linear_coef(self, repn, m.disjunct[i, 1].indicator_var, 1)