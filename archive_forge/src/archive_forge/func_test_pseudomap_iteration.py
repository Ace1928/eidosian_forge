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
def test_pseudomap_iteration(self):
    m = Block()
    m.a = Constraint()
    m.z = Objective()
    m.x = Objective()
    m.v = Objective()
    m.b = Constraint()
    m.t = Block()
    m.s = Block()
    m.c = Constraint()
    m.y = Objective()
    m.w = Objective()
    m.b.deactivate()
    m.z.deactivate()
    m.w.deactivate()
    m.t.deactivate()
    self.assertEqual(['a', 'z', 'x', 'v', 'b', 't', 's', 'c', 'y', 'w'], list(m.component_map()))
    self.assertEqual(['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'], list(m.component_map(set([Constraint, Objective]))))
    self.assertEqual(['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'], list(m.component_map([Constraint, Objective])))
    self.assertEqual(['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'], list(m.component_map([Objective, Constraint])))
    self.assertEqual(['a', 'b', 'c'], list(m.component_map(Constraint)))
    self.assertEqual(['z', 'x', 'v', 'y', 'w'], list(m.component_map(set([Objective]))))
    self.assertEqual(['a', 'x', 'v', 's', 'c', 'y'], list(m.component_map(active=True)))
    self.assertEqual(['a', 'x', 'v', 'c', 'y'], list(m.component_map(set([Constraint, Objective]), active=True)))
    self.assertEqual(['a', 'x', 'v', 'c', 'y'], list(m.component_map([Constraint, Objective], active=True)))
    self.assertEqual(['a', 'x', 'v', 'c', 'y'], list(m.component_map([Objective, Constraint], active=True)))
    self.assertEqual(['a', 'c'], list(m.component_map(Constraint, active=True)))
    self.assertEqual(['x', 'v', 'y'], list(m.component_map(set([Objective]), active=True)))
    self.assertEqual(['z', 'b', 't', 'w'], list(m.component_map(active=False)))
    self.assertEqual(['z', 'b', 'w'], list(m.component_map(set([Constraint, Objective]), active=False)))
    self.assertEqual(['z', 'b', 'w'], list(m.component_map([Constraint, Objective], active=False)))
    self.assertEqual(['z', 'b', 'w'], list(m.component_map([Objective, Constraint], active=False)))
    self.assertEqual(['b'], list(m.component_map(Constraint, active=False)))
    self.assertEqual(['z', 'w'], list(m.component_map(set([Objective]), active=False)))
    self.assertEqual(['a', 'b', 'c', 's', 't', 'v', 'w', 'x', 'y', 'z'], list(m.component_map(sort=True)))
    self.assertEqual(['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'], list(m.component_map(set([Constraint, Objective]), sort=True)))
    self.assertEqual(['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'], list(m.component_map([Constraint, Objective], sort=True)))
    self.assertEqual(['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'], list(m.component_map([Objective, Constraint], sort=True)))
    self.assertEqual(['a', 'b', 'c'], list(m.component_map(Constraint, sort=True)))
    self.assertEqual(['v', 'w', 'x', 'y', 'z'], list(m.component_map(set([Objective]), sort=True)))
    self.assertEqual(['a', 'c', 's', 'v', 'x', 'y'], list(m.component_map(active=True, sort=True)))
    self.assertEqual(['a', 'c', 'v', 'x', 'y'], list(m.component_map(set([Constraint, Objective]), active=True, sort=True)))
    self.assertEqual(['a', 'c', 'v', 'x', 'y'], list(m.component_map([Constraint, Objective], active=True, sort=True)))
    self.assertEqual(['a', 'c', 'v', 'x', 'y'], list(m.component_map([Objective, Constraint], active=True, sort=True)))
    self.assertEqual(['a', 'c'], list(m.component_map(Constraint, active=True, sort=True)))
    self.assertEqual(['v', 'x', 'y'], list(m.component_map(set([Objective]), active=True, sort=True)))
    self.assertEqual(['b', 't', 'w', 'z'], list(m.component_map(active=False, sort=True)))
    self.assertEqual(['b', 'w', 'z'], list(m.component_map(set([Constraint, Objective]), active=False, sort=True)))
    self.assertEqual(['b', 'w', 'z'], list(m.component_map([Constraint, Objective], active=False, sort=True)))
    self.assertEqual(['b', 'w', 'z'], list(m.component_map([Objective, Constraint], active=False, sort=True)))
    self.assertEqual(['b'], list(m.component_map(Constraint, active=False, sort=True)))
    self.assertEqual(['w', 'z'], list(m.component_map(set([Objective]), active=False, sort=True)))