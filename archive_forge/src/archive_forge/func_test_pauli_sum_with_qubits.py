import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('psum, expected_psum', ((cirq.Z(q0) + cirq.Y(q0), cirq.Z(q1) + cirq.Y(q0)), (2 * cirq.X(q0) + 3 * cirq.Y(q2), 2 * cirq.X(q1) + 3 * cirq.Y(q3)), (cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), cirq.X(q1) * cirq.Y(q2) + cirq.Y(q2) * cirq.Z(q3))))
def test_pauli_sum_with_qubits(psum, expected_psum):
    if len(expected_psum.qubits) == len(psum.qubits):
        assert psum.with_qubits(*expected_psum.qubits) == expected_psum
    else:
        with pytest.raises(ValueError, match='number'):
            psum.with_qubits(*expected_psum.qubits)