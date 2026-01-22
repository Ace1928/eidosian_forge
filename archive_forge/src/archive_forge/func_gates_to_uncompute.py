from __future__ import annotations
from collections.abc import Sequence
import typing
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.instruction import Instruction
from .state_preparation import StatePreparation
def gates_to_uncompute(self) -> QuantumCircuit:
    """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            Circuit to take ``self.params`` vector to :math:`|{00\\ldots0}\\rangle`
        """
    return self._stateprep._gates_to_uncompute()