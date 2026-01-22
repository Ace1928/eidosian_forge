import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
def test_msb_demuxer():
    U1 = unitary_group.rvs(4)
    U2 = unitary_group.rvs(4)
    U_full = np.kron([[1, 0], [0, 0]], U1) + np.kron([[0, 0], [0, 1]], U2)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(3)]
    circuit = cirq.Circuit(_msb_demuxer(qubits, U1, U2))
    assert cirq.approx_eq(U_full, circuit.unitary(), atol=1e-09)
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all((isinstance(op.gate, gates) for op in circuit.all_operations()))