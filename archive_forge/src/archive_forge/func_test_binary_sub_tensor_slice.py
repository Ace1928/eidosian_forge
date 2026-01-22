import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_binary_sub_tensor_slice():
    a = slice(None)
    e = Ellipsis
    assert cirq.slice_for_qubits_equal_to([], 0) == (e,)
    assert cirq.slice_for_qubits_equal_to([0], 0) == (0, e)
    assert cirq.slice_for_qubits_equal_to([0], 1) == (1, e)
    assert cirq.slice_for_qubits_equal_to([1], 0) == (a, 0, e)
    assert cirq.slice_for_qubits_equal_to([1], 1) == (a, 1, e)
    assert cirq.slice_for_qubits_equal_to([2], 0) == (a, a, 0, e)
    assert cirq.slice_for_qubits_equal_to([2], 1) == (a, a, 1, e)
    assert cirq.slice_for_qubits_equal_to([0, 1], 0) == (0, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 2], 0) == (a, 0, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 3], 0) == (a, 0, a, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 3], 2) == (a, 0, a, 1, e)
    assert cirq.slice_for_qubits_equal_to([3, 1], 2) == (a, 1, a, 0, e)
    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 1) == (0, 0, 1, e)
    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 2) == (0, 1, 0, e)
    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 4) == (1, 0, 0, e)
    assert cirq.slice_for_qubits_equal_to([0, 1, 2], 5) == (1, 0, 1, e)
    assert cirq.slice_for_qubits_equal_to([0, 2, 1], 5) == (1, 1, 0, e)
    m = np.array([0] * 16).reshape((2, 2, 2, 2))
    for k in range(16):
        m[cirq.slice_for_qubits_equal_to([3, 2, 1, 0], k)] = k
    assert list(m.reshape(16)) == list(range(16))
    assert cirq.slice_for_qubits_equal_to([0], 1, num_qubits=1) == (1,)
    assert cirq.slice_for_qubits_equal_to([1], 0, num_qubits=2) == (a, 0)
    assert cirq.slice_for_qubits_equal_to([1], 0, num_qubits=3) == (a, 0, a)
    assert cirq.slice_for_qubits_equal_to([2], 0, num_qubits=3) == (a, a, 0)