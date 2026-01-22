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
def test_invalid_argument_values(self):
    m = self.m
    m.w = Var(m.t)
    m.y = Var()
    with self.assertRaises(DAE_Error):
        Simulator(m, package='foo')

    def _con(m, i):
        return m.v[i] == m.w[i] ** 2 + m.y
    m.con = Constraint(m.t, rule=_con)
    with self.assertRaises(DAE_Error):
        Simulator(m, package='scipy')
    m.del_component('con')
    m.del_component('con_index')
    m.del_component('w')
    m.del_component('y')