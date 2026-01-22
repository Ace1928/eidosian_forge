import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def diffeq(m, t, i):
    return m.dxdt[t, i] == t * m.x[t, i] ** 2 + m.y ** 2