import fractions
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('val,resolved', [(sympy.pi, np.pi), (sympy.S.NegativeOne, -1), (sympy.S.Half, 0.5), (sympy.S.One, 1)])
def test_value_of_transformed_types(val, resolved):
    _assert_consistent_resolution(val, resolved)