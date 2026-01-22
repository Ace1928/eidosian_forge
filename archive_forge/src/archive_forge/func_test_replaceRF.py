import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_replaceRF(self):
    self.interface.data.all_variables = ComponentSet()
    self.interface.data.truth_models = ComponentMap()
    self.interface.data.ef_outputs = VarList()
    expr = self.interface.model.obj.expr
    new_expr = self.interface.replaceEF(expr)
    self.assertEqual(expr, new_expr)
    expr = self.interface.model.c1.expr
    new_expr = self.interface.replaceEF(expr)
    self.assertIsNot(expr, new_expr)
    self.assertEqual(str(new_expr), 'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')