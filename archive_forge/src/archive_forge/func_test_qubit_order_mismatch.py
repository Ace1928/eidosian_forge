import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_qubit_order_mismatch():
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='are not an ordered subset'):
        _ = cirq.PauliStringPhasor(1j * cirq.X(q0), qubits=[q1])
    with pytest.raises(ValueError, match='are not an ordered subset'):
        _ = cirq.PauliStringPhasor(1j * cirq.X(q0) * cirq.X(q1), qubits=[q1])
    with pytest.raises(ValueError, match='are not an ordered subset'):
        _ = cirq.PauliStringPhasor(1j * cirq.X(q0), qubits=[])
    with pytest.raises(ValueError, match='are not an ordered subset'):
        _ = cirq.PauliStringPhasor(1j * cirq.X(q0) * cirq.X(q1), qubits=[q1, q0])