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
def test_check_viewsumexpression(self):
    m = self.m
    m.p = Param(initialize=5)
    m.mp = Param(initialize=5, mutable=True)
    m.y = Var()
    m.z = Var()
    t = IndexTemplate(m.t)
    e = m.dv[t] + m.y + m.z == m.v[t]
    temp = _check_viewsumexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.SumExpression)
    self.assertIs(type(temp[1].arg(0)), EXPR.Numeric_GetItemExpression)
    self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
    self.assertEqual(-1, temp[1].arg(1).arg(0))
    self.assertIs(m.y, temp[1].arg(1).arg(1))
    self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
    self.assertEqual(-1, temp[1].arg(2).arg(0))
    self.assertIs(m.z, temp[1].arg(2).arg(1))
    e = m.v[t] == m.y + m.dv[t] + m.z
    temp = _check_viewsumexpression(e, 1)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.SumExpression)
    self.assertIs(type(temp[1].arg(0)), EXPR.Numeric_GetItemExpression)
    self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
    self.assertIs(m.y, temp[1].arg(1).arg(1))
    self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
    self.assertIs(m.z, temp[1].arg(2).arg(1))
    e = 5 * m.dv[t] + 5 * m.y - m.z == m.v[t]
    temp = _check_viewsumexpression(e, 0)
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(type(temp[1]), EXPR.DivisionExpression)
    self.assertIs(type(temp[1].arg(0).arg(0)), EXPR.Numeric_GetItemExpression)
    self.assertIs(m.y, temp[1].arg(0).arg(1).arg(1))
    self.assertIs(m.z, temp[1].arg(0).arg(2).arg(1))
    e = 2 + 5 * m.y - m.z == m.v[t]
    temp = _check_viewsumexpression(e, 0)
    self.assertIs(temp, None)