import numpy as np
import pytest
import cirq
def test_destabilizers():
    t = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 1
    assert destabilizers[0] == cirq.DensePauliString('X', coefficient=1)
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _CNOT(t, 0, 1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 2
    assert destabilizers[0] == cirq.DensePauliString('ZI', coefficient=1)
    assert destabilizers[1] == cirq.DensePauliString('IX', coefficient=1)
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _H(t, 1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 2
    assert destabilizers[0] == cirq.DensePauliString('ZI', coefficient=1)
    assert destabilizers[1] == cirq.DensePauliString('IZ', coefficient=1)