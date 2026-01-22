import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
@unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
def test_rejectStep(self):
    self.interface.model.x[1] = 1.5
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.createConstraints()
    self.interface.data.basis_constraint.activate()
    _, _, _ = self.interface.solveModel()
    self.assertEqual(len(self.interface.data.all_variables), len(self.interface.data.previous_model_state))
    self.assertNotEqual(value(self.interface.model.x[0]), 2.0)
    self.assertNotEqual(value(self.interface.model.x[1]), 1.5)
    self.assertNotEqual(value(self.interface.model.z[0]), 5.0)
    self.assertNotEqual(value(self.interface.model.z[1]), 2.5)
    self.assertNotEqual(value(self.interface.model.z[2]), -1.0)
    self.interface.rejectStep()
    self.assertEqual(value(self.interface.model.x[0]), 2.0)
    self.assertEqual(value(self.interface.model.x[1]), 1.5)
    self.assertEqual(value(self.interface.model.z[0]), 5.0)
    self.assertEqual(value(self.interface.model.z[1]), 2.5)
    self.assertEqual(value(self.interface.model.z[2]), -1.0)