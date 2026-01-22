from typing import Tuple, List
from unittest.mock import create_autospec
import cirq
import numpy as np
from pyquil.gates import MEASURE, RX, DECLARE, H, CNOT, I
from pyquil.quilbase import Pragma, Reset
from cirq_rigetti import circuit_transformers as transformers
def test_transform_cirq_circuit_with_explicit_decompose(parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]) -> None:
    """test that a user add a custom circuit decomposition function"""
    parametric_circuit, param_resolvers = parametric_circuit_with_params
    parametric_circuit.append(cirq.I(cirq.GridQubit(0, 0)))
    parametric_circuit.append(cirq.I(cirq.GridQubit(0, 1)))
    parametric_circuit.append(cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), key='m'))
    circuit = cirq.protocols.resolve_parameters(parametric_circuit, param_resolvers[1])

    def decompose_operation(operation: cirq.Operation) -> List[cirq.Operation]:
        operations = [operation]
        if isinstance(operation.gate, cirq.MeasurementGate) and operation.gate.num_qubits() == 1:
            operations.append(cirq.I(operation.qubits[0]))
        return operations
    program, _ = transformers.build(decompose_operation=decompose_operation)(circuit=circuit)
    assert RX(np.pi / 2, 2) in program.instructions, 'executable should contain an RX(pi) 0 instruction'
    assert I(0) in program.instructions, 'executable should contain an I(0) instruction'
    assert I(1) in program.instructions, 'executable should contain an I(1) instruction'
    assert I(2) in program.instructions, 'executable should contain an I(2) instruction'
    assert DECLARE('m0') in program.instructions, 'executable should declare a read out bit'
    assert MEASURE(0, ('m0', 0)) in program.instructions, 'executable should measure the read out bit'