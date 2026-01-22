from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
@pytest.mark.parametrize(['angles', 'num_cnots'], [[[-0.2312, 0.2312, 1.43, -2.2322], 4], [[0, 0, 0, 0], 0], [[0.3, 0.3, 0.3, 0.3], 0], [[0.3, -0.3, 0.3, -0.3], 2], [[0.3, 0.3, -0.3, -0.3], 2], [[-0.3, 0.3, 0.3, -0.3], 4], [[-0.3, 0.3, -0.3, 0.3], 2], [[0.3, -0.3, -0.3, -0.3], 4], [[-0.3, 0.3, -0.3, -0.3], 4]])
def test_middle_multiplexor(angles, num_cnots):
    a, b, c = cirq.LineQubit.range(3)
    eigvals = np.exp(np.array(angles) * np.pi * 1j)
    d = np.diag(np.sqrt(eigvals))
    mid = block_diag(d, d.conj().T)
    circuit_u1u2_mid = cirq.Circuit(_middle_multiplexor_to_ops(a, b, c, eigvals))
    np.testing.assert_almost_equal(mid, circuit_u1u2_mid.unitary(qubits_that_should_be_present=[a, b, c]))
    assert len([cnot for cnot in list(circuit_u1u2_mid.all_operations()) if isinstance(cnot.gate, cirq.CNotPowGate)]) == num_cnots, f'expected {num_cnots} CNOTs got \n {circuit_u1u2_mid} \n {circuit_u1u2_mid.unitary()}'