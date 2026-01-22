from __future__ import annotations
from collections.abc import Sequence
import typing
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.instruction import Instruction
from .state_preparation import StatePreparation
def broadcast_arguments(self, qargs, cargs):
    return self._stateprep.broadcast_arguments(qargs, cargs)