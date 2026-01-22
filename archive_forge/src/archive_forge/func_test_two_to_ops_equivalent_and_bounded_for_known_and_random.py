import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
@pytest.mark.parametrize('max_partial_cz_depth,max_full_cz_depth,effect', [(0, 0, np.eye(4)), (0, 0, np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0j]])), (0, 0, cirq.unitary(cirq.CZ ** 1e-08)), (0.5, 2, cirq.unitary(cirq.CZ ** 0.5)), (1, 1, cirq.unitary(cirq.CZ)), (1, 1, cirq.unitary(cirq.CNOT)), (1, 1, np.array([[1, 0, 0, 1j], [0, 1, 1j, 0], [0, 1j, 1, 0], [1j, 0, 0, 1]]) * np.sqrt(0.5)), (1, 1, np.array([[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 1]]) * np.sqrt(0.5)), (1, 1, np.array([[1, 0, 0, 1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [1j, 0, 0, 1]]) * np.sqrt(0.5)), (1.5, 3, cirq.map_eigenvalues(cirq.unitary(cirq.SWAP), lambda e: e ** 0.5)), (2, 2, cirq.unitary(cirq.SWAP).dot(cirq.unitary(cirq.CZ))), (3, 3, cirq.unitary(cirq.SWAP)), (3, 3, np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0j]]))] + [(1, 2, _random_single_partial_cz_effect()) for _ in range(10)] + [(2, 2, _random_double_full_cz_effect()) for _ in range(10)] + [(2, 3, _random_double_partial_cz_effect()) for _ in range(10)] + [(3, 3, cirq.testing.random_unitary(4)) for _ in range(10)])
def test_two_to_ops_equivalent_and_bounded_for_known_and_random(max_partial_cz_depth, max_full_cz_depth, effect):
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    operations_with_partial = cirq.two_qubit_matrix_to_cz_operations(q0, q1, effect, True)
    operations_with_full = cirq.two_qubit_matrix_to_cz_operations(q0, q1, effect, False)
    assert_ops_implement_unitary(q0, q1, operations_with_partial, effect)
    assert_ops_implement_unitary(q0, q1, operations_with_full, effect)
    assert_cz_depth_below(operations_with_partial, max_partial_cz_depth, False)
    assert_cz_depth_below(operations_with_full, max_full_cz_depth, True)