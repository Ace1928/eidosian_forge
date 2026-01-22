import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_block_derived2(self):

    class Foo(Block):
        pass
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.s = Set(initialize=[1, 2, 3])

    def _block_rule(b, t, s):
        m = b.model()

        def _init(m, j):
            return j * 2
        b.p1 = Param(m.t, default=_init)
        b.v1 = Var(m.t, initialize=5)
    m.foo = Foo(m.t, m.s, rule=_block_rule)
    generate_finite_elements(m.t, 5)
    expand_components(m)
    self.assertEqual(len(m.foo), 18)
    self.assertEqual(len(m.foo[0, 1].p1), 6)
    self.assertEqual(len(m.foo[2, 2].v1), 6)
    self.assertEqual(m.foo[0, 3].p1[6], 12)