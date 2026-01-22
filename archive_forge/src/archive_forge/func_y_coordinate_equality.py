from math import isclose
from sympy.core.numbers import I
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.abc import s, p, a
from sympy.external import import_module
from sympy.physics.control.control_plots import \
from sympy.physics.control.lti import (TransferFunction,
from sympy.testing.pytest import raises, skip
def y_coordinate_equality(plot_data_func, evalf_func, system):
    """Checks whether the y-coordinate value of the plotted
    data point is equal to the value of the function at a
    particular x."""
    x, y = plot_data_func(system)
    x, y = _trim_tuple(x, y)
    y_exp = tuple((evalf_func(system, x_i) for x_i in x))
    return all((Abs(y_exp_i - y_i) < 1e-08 for y_exp_i, y_i in zip(y_exp, y)))