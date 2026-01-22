from __future__ import annotations
from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate
@property
def state_preparation(self) -> QuantumCircuit:
    """The subcircuit implementing the A operator or Hadamards."""
    if self._state_preparation is not None:
        return self._state_preparation
    num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
    hadamards = QuantumCircuit(num_state_qubits, name='H')
    hadamards.h(self.reflection_qubits)
    return hadamards