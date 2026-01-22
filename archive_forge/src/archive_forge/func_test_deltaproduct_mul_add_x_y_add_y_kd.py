from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltaproduct_mul_add_x_y_add_y_kd():
    assert dp((x + y) * (y + KD(i, j)), (j, 1, 3)) == ((x + y) * y) ** 3 + (x + y) * ((x + y) * y) ** 2 * KD(i, 1) + (x + y) * y * (x + y) ** 2 * y * KD(i, 2) + ((x + y) * y) ** 2 * (x + y) * KD(i, 3)
    assert dp((x + y) * (y + KD(i, j)), (j, 1, 1)) == (x + y) * (y + KD(i, 1))
    assert dp((x + y) * (y + KD(i, j)), (j, 2, 2)) == (x + y) * (y + KD(i, 2))
    assert dp((x + y) * (y + KD(i, j)), (j, 3, 3)) == (x + y) * (y + KD(i, 3))
    assert dp((x + y) * (y + KD(i, j)), (j, 1, k)) == ((x + y) * y) ** k + Piecewise((((x + y) * y) ** (-1) * ((x + y) * y) ** i * (x + y) * ((x + y) * y) ** k * ((x + y) * y) ** (-i), (i >= 1) & (i <= k)), (0, True))
    assert dp((x + y) * (y + KD(i, j)), (j, k, 3)) == ((x + y) * y) ** 4 * ((x + y) * y) ** (-k) + Piecewise((((x + y) * y) ** i * ((x + y) * y) ** (-k) * (x + y) * ((x + y) * y) ** 3 * ((x + y) * y) ** (-i), (i >= k) & (i <= 3)), (0, True))
    assert dp((x + y) * (y + KD(i, j)), (j, k, l)) == (x + y) * y * ((x + y) * y) ** l * ((x + y) * y) ** (-k) + Piecewise((((x + y) * y) ** i * ((x + y) * y) ** (-k) * (x + y) * ((x + y) * y) ** l * ((x + y) * y) ** (-i), (i >= k) & (i <= l)), (0, True))