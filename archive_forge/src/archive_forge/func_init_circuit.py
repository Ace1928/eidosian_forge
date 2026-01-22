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
def init_circuit(state: QuantumCircuit | Statevector) -> QuantumCircuit:
    """Initialize state by converting the input to a quantum circuit.

    Args:
        state: The state as quantum circuit or statevector.

    Returns:
        The state as quantum circuit.
    """
    if isinstance(state, QuantumCircuit):
        return state
    if not isinstance(state, Statevector):
        state = Statevector(state)
    qc = QuantumCircuit(state.num_qubits)
    qc.append(Initialize(state), qargs=range(state.num_qubits))
    return qc