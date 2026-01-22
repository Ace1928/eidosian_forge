import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_superoperator():
    cnot = cirq.unitary(cirq.CNOT)
    a, b = cirq.LineQubit.range(2)
    m = cirq.Moment()
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.array([[1.0]]))
    m = cirq.Moment(cirq.I(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.eye(4))
    m = cirq.Moment(cirq.IdentityGate(2).on(a, b))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.eye(16))
    m = cirq.Moment(cirq.S(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.diag([1, -1j, 1j, 1]))
    m = cirq.Moment(cirq.CNOT(a, b))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.kron(cnot, cnot))
    m = cirq.Moment(cirq.depolarize(0.75).on(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2)