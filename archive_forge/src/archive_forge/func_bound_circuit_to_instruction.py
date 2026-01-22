from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.bit import Bit
from qiskit.circuit.library.data_preparation import Initialize
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
def bound_circuit_to_instruction(circuit: QuantumCircuit) -> Instruction:
    """Build an :class:`~qiskit.circuit.Instruction` object from
    a :class:`~qiskit.circuit.QuantumCircuit`

    This is a specialized version of :func:`~qiskit.converters.circuit_to_instruction`
    to avoid deep copy. This requires a quantum circuit whose parameters are all bound.
    Because this does not take a copy of the input circuit, this assumes that the input
    circuit won't be modified.

    If https://github.com/Qiskit/qiskit-terra/issues/7983 is resolved,
    we can remove this function.

    Args:
        circuit(QuantumCircuit): Input quantum circuit

    Returns:
        An :class:`~qiskit.circuit.Instruction` object
    """
    if len(circuit.qregs) > 1:
        return circuit.to_instruction()
    inst = Instruction(name=circuit.name, num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, params=[])
    inst.definition = circuit
    return inst