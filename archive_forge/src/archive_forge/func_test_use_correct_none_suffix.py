from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def test_use_correct_none_suffix(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 111))
    m.b = Block()
    m.b.d = Disjunct()
    m.b.d.foo = Block()
    m.b.d.c = Constraint(expr=m.x >= 9)
    m.b.BigM = Suffix()
    m.b.BigM[None] = 10
    m.b.d.foo.BigM = Suffix()
    m.b.d.foo.BigM[None] = 1
    m.d = Disjunct()
    m.disj = Disjunction(expr=[m.d, m.b.d])
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    cons_list = bigm.get_transformed_constraints(m.b.d.c)
    lb = cons_list[0]
    self.assertEqual(lb.lower, 9)
    self.assertIsNone(lb.upper)
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 10)
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertIs(repn.linear_vars[0], m.x)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertIs(repn.linear_vars[1], m.b.d.binary_indicator_var)
    self.assertEqual(repn.linear_coefs[1], -10)