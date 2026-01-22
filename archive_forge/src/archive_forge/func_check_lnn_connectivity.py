import copy
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix
def check_lnn_connectivity(qc: QuantumCircuit) -> bool:
    """Check that the synthesized circuit qc fits linear nearest neighbor connectivity.

    Args:
        qc: a :class:`.QuantumCircuit` containing only CX and single qubit gates.

    Returns:
        bool: True if the circuit has linear nearest neighbor connectivity.

    Raises:
        CircuitError: if qc has a non-CX two-qubit gate.
    """
    for instruction in qc.data:
        if instruction.operation.num_qubits > 1:
            if instruction.operation.name == 'cx':
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                if dist != 1:
                    return False
            else:
                raise CircuitError('The circuit has two-qubits gates different than CX.')
    return True