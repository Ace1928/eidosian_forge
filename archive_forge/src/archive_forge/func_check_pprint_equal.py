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
def check_pprint_equal(self, m, unpickle):
    m_buf = StringIO()
    m.pprint(ostream=m_buf)
    m_output = m_buf.getvalue()
    unpickle_buf = StringIO()
    unpickle.pprint(ostream=unpickle_buf)
    unpickle_output = unpickle_buf.getvalue()
    self.assertMultiLineEqual(m_output, unpickle_output)