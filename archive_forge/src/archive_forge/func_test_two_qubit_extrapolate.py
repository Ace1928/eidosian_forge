import re
import numpy as np
import pytest
import sympy
import cirq
def test_two_qubit_extrapolate():
    cz2 = cirq.MatrixGate(np.diag([1, 1, 1, 1j]))
    cz4 = cirq.MatrixGate(np.diag([1, 1, 1, (1 + 1j) * np.sqrt(0.5)]))
    i = cirq.MatrixGate(np.eye(4))
    assert cirq.approx_eq(cz2 ** 0, i, atol=1e-09)
    assert cirq.approx_eq(cz4 ** 0, i, atol=1e-09)
    assert cirq.approx_eq(cz2 ** 0.5, cz4, atol=1e-09)
    with pytest.raises(TypeError):
        _ = cz2 ** sympy.Symbol('a')