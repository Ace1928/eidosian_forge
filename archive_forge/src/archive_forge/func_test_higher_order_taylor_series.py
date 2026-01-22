import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.core.expr import polynomial_degree, differentiate
def test_higher_order_taylor_series(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.5)
    m.y = pyo.Var(initialize=1.5)
    e = m.x * m.y
    tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=2)
    for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            m.x.value = _x
            m.y.value = _y
            self.assertAlmostEqual(pyo.value(e), pyo.value(tse))
    e = m.x ** 3 + m.y ** 3
    tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=3)
    for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            m.x.value = _x
            m.y.value = _y
            self.assertAlmostEqual(pyo.value(e), pyo.value(tse))
    e = (m.x * m.y) ** 2
    tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=4)
    for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            m.x.value = _x
            m.y.value = _y
            self.assertAlmostEqual(pyo.value(e), pyo.value(tse))