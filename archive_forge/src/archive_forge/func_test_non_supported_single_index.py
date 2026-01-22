import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
def test_non_supported_single_index(self):
    m = ConcreteModel()
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m = ConcreteModel()
    m.s = ContinuousSet(bounds=(0, 10))
    m.t = ContinuousSet(bounds=(0, 5))
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m = self.m

    def _diffeq(m, t):
        return m.dv[t] == m.v[t] ** 2 + m.v[t]
    m.con1 = Constraint(m.t, rule=_diffeq)
    m.con2 = Constraint(m.t, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')
    m.del_component('con2')
    m = self.m

    def _diffeq(m, t):
        return m.dv[t] == m.dv[t] + m.v[t] ** 2
    m.con1 = Constraint(m.t, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')