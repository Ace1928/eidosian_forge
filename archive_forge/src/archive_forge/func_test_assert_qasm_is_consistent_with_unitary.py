from typing import Tuple
import warnings
import numpy as np
import pytest
import cirq
def test_assert_qasm_is_consistent_with_unitary():
    try:
        import qiskit as _
    except ImportError:
        warnings.warn("Skipped test_assert_qasm_is_consistent_with_unitary because qiskit isn't installed to verify against.")
        return
    cirq.testing.assert_qasm_is_consistent_with_unitary(Fixed(np.array([[1, 0], [0, 1]]), 'z {0}; z {0};'))
    cirq.testing.assert_qasm_is_consistent_with_unitary(Fixed(np.array([[1, 0], [0, -1]]), 'z {0};'))
    with pytest.raises(AssertionError, match='Not equal'):
        cirq.testing.assert_qasm_is_consistent_with_unitary(Fixed(np.array([[1, 0], [0, -1]]), 'x {0};'))
    cirq.testing.assert_qasm_is_consistent_with_unitary(cirq.CNOT)
    cirq.testing.assert_qasm_is_consistent_with_unitary(cirq.CNOT.on(cirq.NamedQubit('a'), cirq.NamedQubit('b')))
    cirq.testing.assert_qasm_is_consistent_with_unitary(cirq.CNOT.on(cirq.NamedQubit('b'), cirq.NamedQubit('a')))
    with pytest.raises(AssertionError, match='QASM not consistent'):
        cirq.testing.assert_qasm_is_consistent_with_unitary(Fixed(np.array([[1, 0], [0, -1]]), 'JUNK$&*@($#::=[];'))
    cirq.testing.assert_qasm_is_consistent_with_unitary(QuditGate())