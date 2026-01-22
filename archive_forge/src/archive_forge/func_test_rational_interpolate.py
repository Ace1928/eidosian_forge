from sympy.polys.polyfuncs import (
from sympy.polys.polyerrors import (
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, d, e, x, y, z
def test_rational_interpolate():
    x, y = symbols('x,y')
    xdata = [1, 2, 3, 4, 5, 6]
    ydata1 = [120, 150, 200, 255, 312, 370]
    ydata2 = [-210, -35, 105, 231, 350, 465]
    assert rational_interpolate(list(zip(xdata, ydata1)), 2) == (60 * x ** 2 + 60) / x
    assert rational_interpolate(list(zip(xdata, ydata1)), 3) == (60 * x ** 2 + 60) / x
    assert rational_interpolate(list(zip(xdata, ydata2)), 2, X=y) == (105 * y ** 2 - 525) / (y + 1)
    xdata = list(range(1, 11))
    ydata = [-1923885361858460, -5212158811973685, -9838050145867125, -15662936261217245, -22469424125057910, -30073793365223685, -38332297297028735, -47132954289530109, -56387719094026320, -66026548943876885]
    assert rational_interpolate(list(zip(xdata, ydata)), 5) == (-12986226192544605 * x ** 4 + 8657484128363070 * x ** 3 - 30301194449270745 * x ** 2 + 4328742064181535 * x - 4328742064181535) / (x ** 3 + 9 * x ** 2 - 3 * x + 11)