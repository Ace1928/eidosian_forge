from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterTable, ParameterView
@qregs.setter
def qregs(self, qregs):
    """Set the quantum registers associated with the circuit."""
    if not self._is_initialized:
        return
    self._qregs = []
    self._ancillas = []
    self._qubit_indices = {}
    self._data = CircuitData(clbits=self._data.clbits)
    self._parameter_table = ParameterTable()
    self.global_phase = 0
    self._is_built = False
    self.add_register(*qregs)