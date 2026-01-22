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
def step_res_tester(sys, expected_value):
    x, y = _to_tuple(*step_response_numerical_data(sys, adaptive=False, nb_of_points=10))
    x_check = check_point_accuracy(x, expected_value[0])
    y_check = check_point_accuracy(y, expected_value[1])
    return x_check and y_check