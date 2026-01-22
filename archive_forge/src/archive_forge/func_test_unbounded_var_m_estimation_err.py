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
def test_unbounded_var_m_estimation_err(self):
    m = models.makeTwoTermDisj_IndexedConstraints()
    self.assertRaisesRegex(GDP_Error, "Cannot estimate M for unbounded expressions.\\n\\t\\(found while processing constraint 'b.simpledisj1.c\\[1\\]'\\). Please specify a value of M or ensure all variables that appear in the constraint are bounded.", TransformationFactory('gdp.bigm').apply_to, m)