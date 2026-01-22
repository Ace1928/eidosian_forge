import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.core.expr import polynomial_degree, differentiate
def test_first_order_taylor_series(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.x.value = 1
    exprs_to_test = [m.x ** 2, pyo.exp(m.x), (m.x + 2) ** 2]
    for e in exprs_to_test:
        tsa = taylor_series_expansion(e)
        self.assertAlmostEqual(pyo.differentiate(e, wrt=m.x), pyo.differentiate(tsa, wrt=m.x))
        self.assertAlmostEqual(pyo.value(e), pyo.value(tsa))
        self.assertEqual(polynomial_degree(tsa), 1)