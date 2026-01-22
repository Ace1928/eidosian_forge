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
def final_measurement_mapping(circuit: QuantumCircuit) -> dict[int, int]:
    """Return the final measurement mapping for the circuit.

    Dict keys label measured qubits, whereas the values indicate the
    classical bit onto which that qubits measurement result is stored.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of qubits to classical bits for final measurements.
    """
    active_qubits = list(range(circuit.num_qubits))
    active_cbits = list(range(circuit.num_clbits))
    mapping = {}
    for item in circuit._data[::-1]:
        if item.operation.name == 'measure':
            cbit = circuit.find_bit(item.clbits[0]).index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[qbit] = cbit
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item.operation.name not in ['barrier', 'delay']:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)
        if not active_cbits or not active_qubits:
            break
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    return mapping