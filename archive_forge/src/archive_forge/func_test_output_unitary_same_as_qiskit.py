import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_output_unitary_same_as_qiskit():
    qubits = tuple(_make_qubits(5))
    operations = _all_operations(*qubits, include_measurements=False)
    output = cirq.QasmOutput(operations, qubits, header='Generated from Cirq', precision=10)
    text = str(output)
    circuit = cirq.Circuit(operations)
    cirq_unitary = circuit.unitary(qubit_order=qubits)
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(text, cirq_unitary)