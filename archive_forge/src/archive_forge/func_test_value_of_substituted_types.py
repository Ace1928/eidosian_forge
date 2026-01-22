import fractions
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('val,resolved', [(sympy.I, 1j)])
def test_value_of_substituted_types(val, resolved):
    _assert_consistent_resolution(val, resolved)