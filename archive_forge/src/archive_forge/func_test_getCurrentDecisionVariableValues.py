import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_getCurrentDecisionVariableValues(self):
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    current_values = self.interface.getCurrentDecisionVariableValues()
    for var in self.interface.decision_variables:
        self.assertIn(var.name, list(current_values.keys()))
        self.assertEqual(current_values[var.name], value(var))