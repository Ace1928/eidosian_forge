import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_hierarchical_blocks(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.t = ContinuousSet(bounds=(0, 10))
    m.b.c = Block()

    def _d_rule(d, t):
        m = d.model()
        d.x = Var()
        return d
    m.b.c.d = Block(m.b.t, rule=_d_rule)
    m.b.y = Var(m.b.t)

    def _con_rule(b, t):
        return b.y[t] <= b.c.d[t].x
    m.b.con = Constraint(m.b.t, rule=_con_rule)
    generate_finite_elements(m.b.t, 5)
    expand_components(m)
    self.assertEqual(len(m.b.c.d), 6)
    self.assertEqual(len(m.b.con), 6)
    self.assertEqual(len(m.b.y), 6)