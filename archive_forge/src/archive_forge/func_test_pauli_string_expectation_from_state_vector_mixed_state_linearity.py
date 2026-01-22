import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pauli_string_expectation_from_state_vector_mixed_state_linearity():
    n_qubits = 6
    state_vector1 = cirq.testing.random_superposition(2 ** n_qubits)
    state_vector2 = cirq.testing.random_superposition(2 ** n_qubits)
    rho1 = np.outer(state_vector1, np.conj(state_vector1))
    rho2 = np.outer(state_vector2, np.conj(state_vector2))
    density_matrix = rho1 / 2 + rho2 / 2
    qubits = cirq.LineQubit.range(n_qubits)
    q_map = {q: i for i, q in enumerate(qubits)}
    paulis = [cirq.X, cirq.Y, cirq.Z]
    pauli_string = cirq.PauliString({q: np.random.choice(paulis) for q in qubits})
    a = pauli_string.expectation_from_state_vector(state_vector1, q_map)
    b = pauli_string.expectation_from_state_vector(state_vector2, q_map)
    c = pauli_string.expectation_from_density_matrix(density_matrix, q_map)
    np.testing.assert_allclose(0.5 * (a + b), c)