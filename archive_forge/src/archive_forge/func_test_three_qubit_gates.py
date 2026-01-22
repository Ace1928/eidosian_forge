from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
@pytest.mark.parametrize('qasm_gate,cirq_gate', three_qubit_gates)
def test_three_qubit_gates(qasm_gate: str, cirq_gate: cirq.testing.TwoQubitGate):
    qasm = f'\n     OPENQASM 2.0;\n     include "qelib1.inc";\n     qreg q1[2];\n     qreg q2[2];\n     qreg q3[2];\n     {qasm_gate} q1[0], q1[1], q2[0];\n     {qasm_gate} q1, q2[0], q3[0];\n     {qasm_gate} q1, q2, q3;\n'
    parser = QasmParser()
    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')
    q3_0 = cirq.NamedQubit('q3_0')
    q3_1 = cirq.NamedQubit('q3_1')
    expected_circuit = Circuit()
    expected_circuit.append(cirq_gate(q1_0, q1_1, q2_0))
    expected_circuit.append(cirq_gate(q1_0, q2_0, q3_0))
    expected_circuit.append(cirq_gate(q1_1, q2_0, q3_0))
    expected_circuit.append(cirq_gate(q1_0, q2_0, q3_0))
    expected_circuit.append(cirq_gate(q1_1, q2_1, q3_1))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2, 'q3': 2}