from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
@m.Disjunct()
def d2(d):
    d.c = Constraint(expr=m.cost == 5)
    d.do_act = Constraint(expr=sum((m.act_time[t] for t in m.Time)) == 1)

    @d.Constraint(m.Time)
    def ms(d, t):
        return t * m.act_time[t] + 2 <= m.makespan