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
def test_reclassify_component(self):
    m = Block()
    m.a = Var()
    m.b = Var()
    m.c = Param()
    self.assertEqual(len(m.component_map(Var)), 2)
    self.assertEqual(len(m.component_map(Param)), 1)
    self.assertEqual(['a', 'b'], list(m.component_map(Var)))
    self.assertEqual(['c'], list(m.component_map(Param)))
    m.reclassify_component_type(m.b, Param)
    self.assertEqual(len(m.component_map(Var)), 1)
    self.assertEqual(len(m.component_map(Param)), 2)
    self.assertEqual(['a'], list(m.component_map(Var)))
    self.assertEqual(['b', 'c'], list(m.component_map(Param)))
    m.reclassify_component_type(m.b, Var)
    self.assertEqual(len(m.component_map(Var)), 2)
    self.assertEqual(len(m.component_map(Param)), 1)
    self.assertEqual(['a', 'b'], list(m.component_map(Var)))
    self.assertEqual(['c'], list(m.component_map(Param)))
    m.reclassify_component_type(m.c, Var)
    self.assertEqual(len(m.component_map(Var)), 3)
    self.assertEqual(len(m.component_map(Param)), 0)
    self.assertTrue(m.contains_component(Var))
    self.assertFalse(m.contains_component(Param))
    self.assertFalse(m.contains_component(Constraint))
    self.assertEqual(['a', 'b', 'c'], list(m.component_map(Var)))
    self.assertEqual([], list(m.component_map(Param)))
    m.reclassify_component_type(m.c, Param)
    self.assertEqual(len(m.component_map(Var)), 2)
    self.assertEqual(len(m.component_map(Param)), 1)
    self.assertEqual(len(m.component_map(Constraint)), 0)
    self.assertTrue(m.contains_component(Var))
    self.assertTrue(m.contains_component(Param))
    self.assertFalse(m.contains_component(Constraint))
    self.assertEqual(['a', 'b'], list(m.component_map(Var)))
    self.assertEqual(['c'], list(m.component_map(Param)))
    m.reclassify_component_type(m.a, Constraint)
    self.assertEqual(len(m.component_map(Var)), 1)
    self.assertEqual(len(m.component_map(Param)), 1)
    self.assertEqual(len(m.component_map(Constraint)), 1)
    self.assertTrue(m.contains_component(Var))
    self.assertTrue(m.contains_component(Param))
    self.assertTrue(m.contains_component(Constraint))
    self.assertEqual(['b'], list(m.component_map(Var)))
    self.assertEqual(['c'], list(m.component_map(Param)))
    self.assertEqual(['a'], list(m.component_map(Constraint)))
    m.reclassify_component_type(m.a, Param)
    m.reclassify_component_type(m.b, Param)
    self.assertEqual(len(m.component_map(Var)), 0)
    self.assertEqual(len(m.component_map(Param)), 3)
    self.assertEqual(len(m.component_map(Constraint)), 0)
    self.assertFalse(m.contains_component(Var))
    self.assertTrue(m.contains_component(Param))
    self.assertFalse(m.contains_component(Constraint))
    self.assertEqual([], list(m.component_map(Var)))
    self.assertEqual(['a', 'b', 'c'], list(m.component_map(Param)))
    self.assertEqual([], list(m.component_map(Constraint)))
    m.reclassify_component_type('b', Var, preserve_declaration_order=False)
    m.reclassify_component_type('c', Var, preserve_declaration_order=False)
    m.reclassify_component_type('a', Var, preserve_declaration_order=False)
    self.assertEqual(len(m.component_map(Var)), 3)
    self.assertEqual(len(m.component_map(Param)), 0)
    self.assertEqual(len(m.component_map(Constraint)), 0)
    self.assertTrue(m.contains_component(Var))
    self.assertFalse(m.contains_component(Param))
    self.assertFalse(m.contains_component(Constraint))
    self.assertEqual(['b', 'c', 'a'], list(m.component_map(Var)))
    self.assertEqual([], list(m.component_map(Param)))
    self.assertEqual([], list(m.component_map(Constraint)))