from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
@classmethod
def from_linear_function(cls, linear_function):
    """Create a Clifford from a Linear Function.

        If the linear function is represented by a nxn binary invertible matrix A,
        then the corresponding Clifford has symplectic matrix [[A^t, 0], [0, A^{-1}]].

        Args:
            linear_function (LinearFunction): A linear function to be converted.

        Returns:
            Clifford: the Clifford object for this linear function.
        """
    from qiskit.synthesis.linear import calc_inverse_matrix
    mat = linear_function.linear
    mat_t = np.transpose(mat)
    mat_i = calc_inverse_matrix(mat)
    dim = len(mat)
    zero = np.zeros((dim, dim), dtype=int)
    symplectic_mat = np.block([[mat_t, zero], [zero, mat_i]])
    phase = np.zeros(2 * dim, dtype=int)
    tableau = cls._stack_table_phase(symplectic_mat, phase)
    return tableau