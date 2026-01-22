import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_noise():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.H(a), cirq.Z(a), cirq.H(a), cirq.measure(a)]), [[0, 0], [0, 1]], atol=0.0001)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.H(a), cirq.Z(a), cirq.H(a), cirq.measure(a)], noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1.0))), [[1, 0], [0, 0]], atol=0.0001)