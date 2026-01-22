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
def check_indexed_xor_constraints(self, transformation):
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    xor = m.component('_pyomo_gdp_%s_reformulation' % transformation).component('disjunction_xor')
    self.assertIsInstance(xor, Constraint)
    for i in m.disjunction.index_set():
        repn = generate_standard_repn(xor[i].body)
        self.assertEqual(repn.constant, 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, m.disjunction[i].disjuncts[0].indicator_var, 1)
        check_linear_coef(self, repn, m.disjunction[i].disjuncts[1].indicator_var, 1)
        self.assertEqual(xor[i].lower, 1)
        self.assertEqual(xor[i].upper, 1)