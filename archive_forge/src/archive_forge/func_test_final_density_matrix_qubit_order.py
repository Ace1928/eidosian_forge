import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_qubit_order():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[a, b]), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.5j], [0, 0, -0.5j, 0.5]])
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[b, a]), [[0, 0, 0, 0], [0, 0.5, 0, 0.5j], [0, 0, 0, 0], [0, -0.5j, 0, 0.5]])
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[b, a], noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1.0))), [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])