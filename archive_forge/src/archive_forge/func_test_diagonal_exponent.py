import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_diagonal_exponent():
    diagonal_angles = [2, 3, 5, 7, 11, 13, 17, 19]
    diagonal_gate = cirq.ThreeQubitDiagonalGate(diagonal_angles)
    sqrt_diagonal_gate = diagonal_gate ** 0.5
    expected_angles = [prime / 2 for prime in diagonal_angles]
    np.testing.assert_allclose(expected_angles, sqrt_diagonal_gate._diag_angles_radians, atol=1e-08)
    assert cirq.pow(cirq.ThreeQubitDiagonalGate(diagonal_angles), 'test', None) is None