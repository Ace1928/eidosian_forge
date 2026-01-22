from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_deduplicate_component_data_iterindex(self):
    m = ConcreteModel()
    m.b = Block()
    m.x = Var()
    m.z_x = Reference(m.x)
    m.I = Var([1, 3, 2])
    m.z_I = Reference(m.I)
    m.b.y = Var()
    m.z_y = Reference(m.b.y)
    m.b.J = Var([4, 6, 5])
    m.z_J = Reference(m.b.J)
    m.c = Block([2, 1])
    m.c[1].A = Var([(0, 2), (1, 1)])
    m.c[2].A = Var([(0, 3), (1, 1)])
    m.z_AA = Reference(m.c[:].A[1, :])
    m.z_A = Reference(m.c[:].A[1, :])
    ans = list(m.component_data_iterindex(Var))
    self.assertEqual(ans, [(('x', None), m.x), (('I', 1), m.I[1]), (('I', 3), m.I[3]), (('I', 2), m.I[2]), (('z_y', None), m.b.y), (('z_J', 4), m.b.J[4]), (('z_J', 6), m.b.J[6]), (('z_J', 5), m.b.J[5]), (('z_AA', (2, 1)), m.c[2].A[1, 1]), (('z_AA', (1, 1)), m.c[1].A[1, 1]), (('A', (0, 3)), m.c[2].A[0, 3]), (('A', (0, 2)), m.c[1].A[0, 2])])
    ans = list(m.component_data_iterindex(Var, sort=SortComponents.SORTED_INDICES))
    self.assertEqual(ans, [(('x', None), m.x), (('I', 1), m.I[1]), (('I', 2), m.I[2]), (('I', 3), m.I[3]), (('z_y', None), m.b.y), (('z_J', 4), m.b.J[4]), (('z_J', 5), m.b.J[5]), (('z_J', 6), m.b.J[6]), (('z_AA', (1, 1)), m.c[1].A[1, 1]), (('z_AA', (2, 1)), m.c[2].A[1, 1]), (('A', (0, 2)), m.c[1].A[0, 2]), (('A', (0, 3)), m.c[2].A[0, 3])])
    ans = list(m.component_data_iterindex(Var, sort=SortComponents.ALPHABETICAL))
    self.assertEqual(ans, [(('I', 1), m.I[1]), (('I', 3), m.I[3]), (('I', 2), m.I[2]), (('x', None), m.x), (('z_A', (2, 1)), m.c[2].A[1, 1]), (('z_A', (1, 1)), m.c[1].A[1, 1]), (('z_J', 4), m.b.J[4]), (('z_J', 6), m.b.J[6]), (('z_J', 5), m.b.J[5]), (('z_y', None), m.b.y), (('A', (0, 3)), m.c[2].A[0, 3]), (('A', (0, 2)), m.c[1].A[0, 2])])
    ans = list(m.component_data_iterindex(Var, sort=SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES))
    self.assertEqual(ans, [(('I', 1), m.I[1]), (('I', 2), m.I[2]), (('I', 3), m.I[3]), (('x', None), m.x), (('z_A', (1, 1)), m.c[1].A[1, 1]), (('z_A', (2, 1)), m.c[2].A[1, 1]), (('z_J', 4), m.b.J[4]), (('z_J', 5), m.b.J[5]), (('z_J', 6), m.b.J[6]), (('z_y', None), m.b.y), (('A', (0, 2)), m.c[1].A[0, 2]), (('A', (0, 3)), m.c[2].A[0, 3])])