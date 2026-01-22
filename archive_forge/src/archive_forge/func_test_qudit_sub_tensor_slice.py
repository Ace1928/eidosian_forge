import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_qudit_sub_tensor_slice():
    a = slice(None)
    sfqet = cirq.slice_for_qubits_equal_to
    assert sfqet([], 0, qid_shape=()) == ()
    assert sfqet([0], 0, qid_shape=(3,)) == (0,)
    assert sfqet([0], 1, qid_shape=(3,)) == (1,)
    assert sfqet([0], 2, qid_shape=(3,)) == (2,)
    assert sfqet([2], 0, qid_shape=(1, 2, 3)) == (a, a, 0)
    assert sfqet([2], 2, qid_shape=(1, 2, 3)) == (a, a, 2)
    assert sfqet([2], big_endian_qureg_value=2, qid_shape=(1, 2, 3)) == (a, a, 2)
    assert sfqet([1, 3], 3 * 2 + 1, qid_shape=(2, 3, 4, 5)) == (a, 1, a, 2)
    assert sfqet([3, 1], 5 * 2 + 1, qid_shape=(2, 3, 4, 5)) == (a, 2, a, 1)
    assert sfqet([2, 1, 0], 9 * 2 + 3 * 1, qid_shape=(3,) * 3) == (2, 1, 0)
    assert sfqet([1, 3], big_endian_qureg_value=5 * 1 + 2, qid_shape=(2, 3, 4, 5)) == (a, 1, a, 2)
    assert sfqet([3, 1], big_endian_qureg_value=3 * 1 + 2, qid_shape=(2, 3, 4, 5)) == (a, 2, a, 1)
    m = np.array([0] * 24).reshape((1, 2, 3, 4))
    for k in range(24):
        m[sfqet([3, 2, 1, 0], k, qid_shape=(1, 2, 3, 4))] = k
    assert list(m.reshape(24)) == list(range(24))
    assert sfqet([0], 1, num_qubits=1, qid_shape=(3,)) == (1,)
    assert sfqet([1], 0, num_qubits=3, qid_shape=(3, 3, 3)) == (a, 0, a)
    with pytest.raises(ValueError, match='len.* !='):
        sfqet([], num_qubits=2, qid_shape=(1, 2, 3))
    with pytest.raises(ValueError, match='exactly one'):
        sfqet([0, 1, 2], 5, big_endian_qureg_value=5)