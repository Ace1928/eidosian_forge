import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
def test_simpleInequality1(self):
    m = self.m
    e = m.a < m.b
    self.assertIs(type(e), InequalityExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.a)
    self.assertIs(e.arg(1), m.b)
    self.assertEqual(e._strict, True)
    e = m.a <= m.b
    self.assertIs(type(e), InequalityExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.a)
    self.assertIs(e.arg(1), m.b)
    self.assertEqual(e._strict, False)
    e = m.a > m.b
    self.assertIs(type(e), InequalityExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.b)
    self.assertIs(e.arg(1), m.a)
    self.assertEqual(e._strict, True)
    e = m.a >= m.b
    self.assertIs(type(e), InequalityExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.b)
    self.assertIs(e.arg(1), m.a)
    self.assertEqual(e._strict, False)