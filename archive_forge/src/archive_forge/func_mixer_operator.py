from __future__ import annotations
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
@mixer_operator.setter
def mixer_operator(self, mixer_operator) -> None:
    """Sets mixer operator.

        Args:
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): mixer
                operator or circuit to set.
        """
    self._mixer = mixer_operator
    self._invalidate()