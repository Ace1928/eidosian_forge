from __future__ import annotations
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
Return the ordered PauliList for the n-qubit Pauli basis.

    Args:
        num_qubits (int): number of qubits
        weight (bool): if True optionally return the basis sorted by Pauli weight
                       rather than lexicographic order (Default: False)

    Returns:
        PauliList: the Paulis for the basis
    