import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_vars_single(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.t2 = ContinuousSet(initialize=[1, 2, 3])
    m.s = Set(initialize=[1, 2, 3])
    m.v1 = Var(m.t, initialize=3)
    m.v2 = Var(m.t, bounds=(4, 10), initialize={0: 2, 10: 12})

    def _init(m, i):
        return i
    m.v3 = Var(m.t, bounds=(-5, 5), initialize=_init)
    m.v4 = Var(m.s, initialize=7, dense=True)
    m.v5 = Var(m.t2, dense=True)
    expansion_map = ComponentMap()
    generate_finite_elements(m.t, 5)
    update_contset_indexed_component(m.v1, expansion_map)
    update_contset_indexed_component(m.v2, expansion_map)
    update_contset_indexed_component(m.v3, expansion_map)
    update_contset_indexed_component(m.v4, expansion_map)
    update_contset_indexed_component(m.v5, expansion_map)
    self.assertTrue(len(m.v1) == 6)
    self.assertTrue(len(m.v2) == 6)
    self.assertTrue(len(m.v3) == 6)
    self.assertTrue(len(m.v4) == 3)
    self.assertTrue(len(m.v5) == 3)
    self.assertTrue(value(m.v1[2]) == 3)
    self.assertTrue(m.v1[4].ub is None)
    self.assertTrue(m.v1[6].lb is None)
    self.assertTrue(m.v2[2].value is None)
    self.assertTrue(m.v2[4].lb == 4)
    self.assertTrue(m.v2[8].ub == 10)
    self.assertTrue(value(m.v2[0]) == 2)
    self.assertTrue(value(m.v3[2]) == 2)
    self.assertTrue(m.v3[4].lb == -5)
    self.assertTrue(m.v3[6].ub == 5)
    self.assertTrue(value(m.v3[8]) == 8)