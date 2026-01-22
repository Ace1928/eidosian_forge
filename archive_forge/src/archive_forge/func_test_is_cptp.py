import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_cptp():
    rt2 = np.sqrt(0.5)
    assert cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, rt2]]), np.array([[0, rt2], [0, 0]])])
    assert cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, 1]]) * 0.5, np.array([[0, 1], [1, 0]]) * 0.5, np.array([[0, -1j], [1j, 0]]) * 0.5, np.array([[1, 0], [0, -1]]) * 0.5])
    assert not cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, 1]]), np.array([[0, 1], [0, 0]])])
    assert not cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])])
    one_qubit_u = cirq.testing.random_unitary(8)
    one_qubit_kraus = np.reshape(one_qubit_u[:, :2], (-1, 2, 2))
    assert cirq.is_cptp(kraus_ops=one_qubit_kraus)
    two_qubit_u = cirq.testing.random_unitary(64)
    two_qubit_kraus = np.reshape(two_qubit_u[:, :4], (-1, 4, 4))
    assert cirq.is_cptp(kraus_ops=two_qubit_kraus)