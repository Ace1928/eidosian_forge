from __future__ import annotations
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
@parameter_bounds.setter
def parameter_bounds(self, bounds: list[tuple[float | None, float | None]] | None) -> None:
    """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
    self._bounds = bounds