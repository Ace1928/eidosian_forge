import re
import numpy as np
import pytest
import sympy
import cirq
def test_matrixgate_unitary_tolerance():
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[1, 0], [0, -0.6]]), unitary_check_atol=0.5)
    _ = cirq.MatrixGate(np.array([[1, 0], [0, 1]]), unitary_check_atol=1)
    _ = cirq.MatrixGate(np.array([[1, 0], [0, -0.6]]), unitary_check_rtol=1)
    _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_atol=0.5)
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_atol=1e-10)
    with pytest.raises(ValueError):
        _ = cirq.MatrixGate(np.array([[0.707, 0.707], [-0.707, 0.707]]), unitary_check_rtol=1e-10)