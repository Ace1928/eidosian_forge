from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltaproduct_mul_add_x_y_kd():
    assert dp((x + y) * KD(i, j), (j, 1, 3)) == 0
    assert dp((x + y) * KD(i, j), (j, 1, 1)) == (x + y) * KD(i, 1)
    assert dp((x + y) * KD(i, j), (j, 2, 2)) == (x + y) * KD(i, 2)
    assert dp((x + y) * KD(i, j), (j, 3, 3)) == (x + y) * KD(i, 3)
    assert dp((x + y) * KD(i, j), (j, 1, k)) == (x + y) * KD(i, 1) * KD(k, 1) + KD(k, 0)
    assert dp((x + y) * KD(i, j), (j, k, 3)) == (x + y) * KD(i, 3) * KD(k, 3) + KD(k, 4)
    assert dp((x + y) * KD(i, j), (j, k, l)) == (x + y) * KD(i, l) * KD(k, l) + KD(k, l + 1)