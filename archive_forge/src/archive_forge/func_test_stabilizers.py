import numpy as np
import pytest
import cirq
def test_stabilizers():
    t = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 1
    assert stabilizers[0] == cirq.DensePauliString('Z', coefficient=-1)
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _CNOT(t, 0, 1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 2
    assert stabilizers[0] == cirq.DensePauliString('XX', coefficient=1)
    assert stabilizers[1] == cirq.DensePauliString('ZZ', coefficient=1)
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _H(t, 1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 2
    assert stabilizers[0] == cirq.DensePauliString('XI', coefficient=1)
    assert stabilizers[1] == cirq.DensePauliString('IX', coefficient=1)