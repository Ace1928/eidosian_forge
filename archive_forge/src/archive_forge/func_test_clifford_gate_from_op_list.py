import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_from_op_list():
    qubit = cirq.NamedQubit('test')
    gate = cirq.CliffordGate.from_op_list([cirq.X(qubit), cirq.Z(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y
    gate = cirq.CliffordGate.from_op_list([cirq.Z(qubit), cirq.X(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y
    gate = cirq.CliffordGate.from_op_list([cirq.X(qubit), cirq.Y(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Z
    gate = cirq.CliffordGate.from_op_list([cirq.Z(qubit), cirq.X(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y
    qubits = cirq.LineQubit.range(2)
    gate = cirq.CliffordGate.from_op_list([cirq.H(qubits[1]), cirq.CZ(*qubits), cirq.H(qubits[1])], qubits)
    assert gate == cirq.CliffordGate.CNOT
    gate = cirq.CliffordGate.from_op_list([cirq.H(qubits[1]), cirq.CNOT(*qubits), cirq.H(qubits[1])], qubits)
    assert gate == cirq.CliffordGate.CZ
    gate = cirq.CliffordGate.from_op_list([cirq.H(qubits[0]), cirq.CZ(qubits[1], qubits[0]), cirq.H(qubits[0])], qubits)
    assert gate != cirq.CliffordGate.CNOT
    gate = cirq.CliffordGate.from_op_list([cirq.H(qubits[0]), cirq.CZ(qubits[1], qubits[0]), cirq.H(qubits[0])], qubits[::-1])
    assert gate == cirq.CliffordGate.CNOT
    with pytest.raises(ValueError, match='only be constructed from the operations that has stabilizer effect'):
        cirq.CliffordGate.from_op_list([cirq.T(qubit)], [qubit])