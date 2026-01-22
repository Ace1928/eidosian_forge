import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.contrib.interior_point.inverse_reduced_hessian import (
@unittest.skipIf(not numdiff_available, 'numdifftools missing')
@unittest.skipIf(not pandas_available, 'pandas missing')
def test_with_binding_constraint(self):
    """there is a binding constraint"""
    model = self._simple_model(add_constraint=True)
    status, H_inv_red_hess = inv_reduced_hessian_barrier(model, [model.b0, model.b['tofu'], model.b['chard']])
    print('test_with_binding_constraint should see an error raised.')