import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_interface_construct(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    scalar_vars = interface.get_scalar_variables()
    self.assertEqual(len(scalar_vars), 1)
    self.assertIs(scalar_vars[0], m.scalar)
    dae_vars = interface.get_indexed_variables()
    self.assertEqual(len(dae_vars), 3)
    dae_var_set = set((self._hashRef(var) for var in dae_vars))
    pred_dae_var = [pyo.Reference(m.var[:, 'A']), pyo.Reference(m.var[:, 'B']), m.input]
    for var in pred_dae_var:
        self.assertIn(self._hashRef(var), dae_var_set)
    dae_expr = interface.get_indexed_expressions()
    dae_expr_set = set((self._hashRef(expr) for expr in dae_expr))
    self.assertEqual(len(dae_expr), 2)
    pred_dae_expr = [pyo.Reference(m.var_squared[:, 'A']), pyo.Reference(m.var_squared[:, 'B'])]
    for expr in pred_dae_expr:
        self.assertIn(self._hashRef(expr), dae_expr_set)