import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_decompose_invalid_qubits():
    qs = cirq.LineQubit.range(3)
    with pytest.raises(ValueError):
        cirq.protocols.decompose_once_with_qubits(cirq.PhasedISwapPowGate(), qs)