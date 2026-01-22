import fractions
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('val', [None, 3.2, np.float32(3.2), int(1), np.int32(45), np.float64(6.3), np.int32(2), np.complex64(1j), np.complex128(2j), complex(1j), fractions.Fraction(3, 2)])
def test_value_of_pass_through_types(val):
    _assert_consistent_resolution(val, val)