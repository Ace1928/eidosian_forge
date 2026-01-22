import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_arcfcn_to_string(self):
    m = ConcreteModel()
    m.x = Var()
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(expression_to_string(asin(m.x), tc, smap=smap), ('arcsin(x1)', False))
    self.assertEqual(expression_to_string(acos(m.x), tc, smap=smap), ('arccos(x1)', False))
    self.assertEqual(expression_to_string(atan(m.x), tc, smap=smap), ('arctan(x1)', False))
    with self.assertRaisesRegex(RuntimeError, 'GAMS files cannot represent the unary function asinh'):
        expression_to_string(asinh(m.x), tc, smap=smap)
    with self.assertRaisesRegex(RuntimeError, 'GAMS files cannot represent the unary function acosh'):
        expression_to_string(acosh(m.x), tc, smap=smap)
    with self.assertRaisesRegex(RuntimeError, 'GAMS files cannot represent the unary function atanh'):
        expression_to_string(atanh(m.x), tc, smap=smap)