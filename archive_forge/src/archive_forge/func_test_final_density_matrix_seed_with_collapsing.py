import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_seed_with_collapsing():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=123, ignore_measurement_results=False), [[0, 0], [0, 1]], atol=0.0001)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=124, ignore_measurement_results=False), [[1, 0], [0, 0]], atol=0.0001)