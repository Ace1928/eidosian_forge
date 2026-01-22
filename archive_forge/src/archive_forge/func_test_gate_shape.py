from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_gate_shape():

    class ShapeGate(cirq.Gate):

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class QubitGate(cirq.Gate):

        def _num_qubits_(self):
            return 3

    class DeprecatedGate(cirq.Gate):

        def num_qubits(self):
            return 3
    shape_gate = ShapeGate()
    assert cirq.qid_shape(shape_gate) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_gate) == 4
    assert shape_gate.num_qubits() == 4
    qubit_gate = QubitGate()
    assert cirq.qid_shape(qubit_gate) == (2, 2, 2)
    assert cirq.num_qubits(qubit_gate) == 3
    assert qubit_gate.num_qubits() == 3
    dep_gate = DeprecatedGate()
    assert cirq.qid_shape(dep_gate) == (2, 2, 2)
    assert cirq.num_qubits(dep_gate) == 3
    assert dep_gate.num_qubits() == 3