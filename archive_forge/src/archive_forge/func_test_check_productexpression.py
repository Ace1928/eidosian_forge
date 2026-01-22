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
def test_check_productexpression(self):
    m = self.m
    m.p = Param(initialize=5)
    m.mp = Param(initialize=5, mutable=True)
    m.y = Var()
    m.z = Var()
    t = IndexTemplate(m.t)
    e = 5 * m.dv[t] == m.v[t]
    temp = _check_productexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    e = m.v[t] == 5 * m.dv[t]
    temp = _check_productexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    e = m.p * m.dv[t] == m.v[t]
    temp = _check_productexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    e = m.v[t] == m.p * m.dv[t]
    temp = _check_productexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    e = m.mp * m.dv[t] == m.v[t]
    temp = _check_productexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(m.mp, temp[1].arg(1))
    self.assertIs(e.arg(1), temp[1].arg(0))
    e = m.v[t] == m.mp * m.dv[t]
    temp = _check_productexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(m.mp, temp[1].arg(1))
    self.assertIs(e.arg(0), temp[1].arg(0))
    e = m.y * m.dv[t] / m.z == m.v[t]
    temp = _check_productexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(e.arg(1), temp[1].arg(0).arg(0))
    self.assertIs(m.z, temp[1].arg(0).arg(1))
    e = m.v[t] == m.y * m.dv[t] / m.z
    temp = _check_productexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(e.arg(0), temp[1].arg(0).arg(0))
    self.assertIs(m.z, temp[1].arg(0).arg(1))
    e = m.y / (m.dv[t] * m.z) == m.mp
    temp = _check_productexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(m.y, temp[1].arg(0))
    self.assertIs(e.arg(1), temp[1].arg(1).arg(0))
    e = m.mp == m.y / (m.dv[t] * m.z)
    temp = _check_productexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(m.y, temp[1].arg(0))
    self.assertIs(e.arg(0), temp[1].arg(1).arg(0))
    e = m.v[t] * m.y / m.z == m.v[t] * m.y / m.z
    temp = _check_productexpression(e, 0)
    self.assertIsNone(temp)
    temp = _check_productexpression(e, 1)
    self.assertIsNone(temp)