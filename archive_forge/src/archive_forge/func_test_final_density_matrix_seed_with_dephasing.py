import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_density_matrix_seed_with_dephasing():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=123), [[0.5 + 0j, 0.0 + 0j], [0.0 + 0j, 0.5 + 0j]], atol=0.0001)
    np.testing.assert_allclose(cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=124), [[0.5 + 0j, 0.0 + 0j], [0.0 + 0j, 0.5 + 0j]], atol=0.0001)