from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
def group_qubit_wise_commuting(self) -> list[PauliList]:
    """Partition a PauliList into sets of mutually qubit-wise commuting Pauli strings.

        Returns:
            list[PauliList]: List of PauliLists where each PauliList contains commutable Pauli operators.
        """
    return self.group_commuting(qubit_wise=True)