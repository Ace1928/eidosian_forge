import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
def test_invalidIndexing(self):
    m = AbstractModel()
    m.A = Set()
    m.p = Param(m.A, mutable=True)
    m.x = Var(m.A)
    m.z = Var()
    try:
        m.p * 2
        self.fail('Expected m.p*2 to raise a TypeError')
    except TypeError:
        pass
    try:
        m.x * 2
        self.fail('Expected m.x*2 to raise a TypeError')
    except TypeError:
        pass
    try:
        2 * m.p
        self.fail('Expected 2*m.p to raise a TypeError')
    except TypeError:
        pass
    try:
        2 * m.x
        self.fail('Expected 2*m.x to raise a TypeError')
    except TypeError:
        pass
    try:
        m.z * m.p
        self.fail('Expected m.z*m.p to raise a TypeError')
    except TypeError:
        pass
    except ValueError:
        pass
    try:
        m.z * m.x
        self.fail('Expected m.z*m.x to raise a TypeError')
    except TypeError:
        pass