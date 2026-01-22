from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def check_bound_constraints_on_disjunctionBlock(self, varlb, varub, disvar, indvar, lb, ub):
    self.assertIsNone(varlb.lower)
    self.assertEqual(varlb.upper, 0)
    repn = generate_standard_repn(varlb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, lb)
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, indvar, -lb)
    ct.check_linear_coef(self, repn, disvar, -1)
    self.assertIsNone(varub.lower)
    self.assertEqual(varub.upper, 0)
    repn = generate_standard_repn(varub.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, -ub)
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, indvar, ub)
    ct.check_linear_coef(self, repn, disvar, 1)