from __future__ import annotations
import typing
from collections.abc import Callable, Sequence
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from .n_local import NLocal
from ..standard_gates import (
def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int) -> Sequence[Sequence[int]]:
    """Overloading to handle the special case of 1 qubit where the entanglement are ignored."""
    if self.num_qubits <= 1:
        return []
    return super().get_entangler_map(rep_num, block_num, num_block_qubits)