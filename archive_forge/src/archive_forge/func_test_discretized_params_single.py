import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_discretized_params_single(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=[(1, 1), (2, 2)])
    m.p1 = Param(m.t, initialize=1)
    m.p2 = Param(m.t, default=2)
    m.p3 = Param(m.t, initialize=1, default=2)

    def _rule1(m, i):
        return i ** 2

    def _rule2(m, i):
        return 2 * i
    m.p4 = Param(m.t, initialize={0: 5, 10: 5}, default=_rule1)
    m.p5 = Param(m.t, initialize=_rule1, default=_rule2)
    generate_finite_elements(m.t, 5)
    with self.assertRaises(ValueError):
        for i in m.t:
            m.p1[i]
    for i in m.t:
        self.assertEqual(m.p2[i], 2)
        if i == 0 or i == 10:
            self.assertEqual(m.p3[i], 1)
            self.assertEqual(m.p4[i], 5)
            self.assertEqual(m.p5[i], i ** 2)
        else:
            self.assertEqual(m.p3[i], 2)
            self.assertEqual(m.p4[i], i ** 2)
            self.assertEqual(m.p5[i], 2 * i)