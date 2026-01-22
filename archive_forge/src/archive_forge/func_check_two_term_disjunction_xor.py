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
def check_two_term_disjunction_xor(self, xor, disj1, disj2):
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(len(xor), 1)
    assertExpressionsEqual(self, xor.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, disj1.binary_indicator_var)), EXPR.MonomialTermExpression((1, disj2.binary_indicator_var))]))
    self.assertEqual(xor.lower, 1)
    self.assertEqual(xor.upper, 1)