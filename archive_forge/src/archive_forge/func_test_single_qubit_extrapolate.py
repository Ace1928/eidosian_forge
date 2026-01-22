import re
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_extrapolate():
    i = cirq.MatrixGate(np.eye(2))
    x = cirq.MatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = cirq.MatrixGate(np.array([[1, 1j], [1j, 1]]) * (1 - 1j) / 2)
    assert cirq.has_unitary(x2)
    x2i = cirq.MatrixGate(np.conj(cirq.unitary(x2).T))
    assert cirq.approx_eq(x ** 0, i, atol=1e-09)
    assert cirq.approx_eq(x2 ** 0, i, atol=1e-09)
    assert cirq.approx_eq(x2 ** 2, x, atol=1e-09)
    assert cirq.approx_eq(x2 ** (-1), x2i, atol=1e-09)
    assert cirq.approx_eq(x2 ** 3, x2i, atol=1e-09)
    assert cirq.approx_eq(x ** (-1), x, atol=1e-09)
    z2 = cirq.MatrixGate(np.array([[1, 0], [0, 1j]]))
    z4 = cirq.MatrixGate(np.array([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]]))
    assert cirq.approx_eq(z2 ** 0.5, z4, atol=1e-09)
    with pytest.raises(TypeError):
        _ = x ** sympy.Symbol('a')