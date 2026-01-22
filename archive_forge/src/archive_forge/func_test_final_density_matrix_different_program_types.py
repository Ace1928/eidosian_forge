import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_different_program_types():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.final_density_matrix(cirq.X), [[0, 0], [0, 1]], atol=1e-08)
    ops = [cirq.H(a), cirq.CNOT(a, b)]
    np.testing.assert_allclose(cirq.final_density_matrix(cirq.Circuit(ops)), [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], atol=1e-08)