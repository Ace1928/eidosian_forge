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
def test_initializeProblem(self):
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    objective, feasibility = self.interface.initializeProblem()
    for var in self.interface.decision_variables:
        self.assertIn(var.name, list(self.interface.initial_decision_bounds.keys()))
        self.assertEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])
    self.assertAlmostEqual(objective, 5.150744273013601)
    self.assertAlmostEqual(feasibility, 0.09569982275514467)
    self.assertTrue(self.interface.data.sm_constraint_basis.active)
    self.assertFalse(self.interface.data.basis_constraint.active)