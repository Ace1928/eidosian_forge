from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def create_nested_structure(self):
    """
        Creates a two-term Disjunction with on nested two-term Disjunction on
        the first Disjunct
        """
    m = ConcreteModel()
    m.outer_d1 = Disjunct()
    m.outer_d1.inner_d1 = Disjunct()
    m.outer_d1.inner_d2 = Disjunct()
    m.outer_d1.inner = Disjunction(expr=[m.outer_d1.inner_d1, m.outer_d1.inner_d2])
    m.outer_d2 = Disjunct()
    m.outer = Disjunction(expr=[m.outer_d1, m.outer_d2])
    return m