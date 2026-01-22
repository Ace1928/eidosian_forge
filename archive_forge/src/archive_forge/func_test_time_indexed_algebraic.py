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
@unittest.skipIf(not scipy_available, 'Scipy is not available')
def test_time_indexed_algebraic(self):
    m = self.m
    m.a = Var(m.t)

    def _diffeq(m, t):
        return m.dv[t] == m.v[t] ** 2 + m.a[t]
    m.con = Constraint(m.t, rule=_diffeq)
    mysim = Simulator(m)
    t = IndexTemplate(m.t)
    self.assertEqual(len(mysim._algvars), 1)
    self.assertTrue(_GetItemIndexer(m.a[t]) in mysim._algvars)
    self.assertEqual(len(mysim._alglist), 0)
    m.del_component('con')