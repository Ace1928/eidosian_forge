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
def test_tuple_wrong_length_err(self):
    m = models.makeTwoTermDisj()
    M = (-20, 19, 32)
    self.assertRaisesRegex(GDP_Error, 'Big-M \\(-20, 19, 32\\) for constraint d\\[0\\].c is not of length two. Expected either a single value or tuple or list of length two*', TransformationFactory('gdp.bigm').apply_to, m, bigM={None: M})