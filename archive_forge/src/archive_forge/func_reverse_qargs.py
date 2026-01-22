from __future__ import annotations
import copy
import re
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit._accelerate.pauli_expval import (
def reverse_qargs(self) -> Statevector:
    """Return a Statevector with reversed subsystem ordering.

        For a tensor product state this is equivalent to reversing the order
        of tensor product subsystems. For a statevector
        :math:`|\\psi \\rangle = |\\psi_{n-1} \\rangle \\otimes ... \\otimes |\\psi_0 \\rangle`
        the returned statevector will be
        :math:`|\\psi_{0} \\rangle \\otimes ... \\otimes |\\psi_{n-1} \\rangle`.

        Returns:
            Statevector: the Statevector with reversed subsystem order.
        """
    ret = copy.copy(self)
    axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
    ret._data = np.reshape(np.transpose(np.reshape(self.data, self._op_shape.tensor_shape), axes), self._op_shape.shape)
    ret._op_shape = self._op_shape.reverse()
    return ret