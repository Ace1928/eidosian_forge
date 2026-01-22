import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_expectation_from_density_matrix_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=complex) / np.sqrt(2)
    rho = np.kron(wf, wf).reshape((8, 8))
    for state in [rho, rho.reshape((2, 2, 2, 2, 2, 2))]:
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}), 1)
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}), 1)
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 1, q1: 0, q2: 2}), 0)
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 1, q1: 2, q2: 0}), 0)
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 2, q1: 0, q2: 1}), -1)
        np.testing.assert_allclose(z.expectation_from_density_matrix(state, {q0: 2, q1: 1, q2: 0}), -1)