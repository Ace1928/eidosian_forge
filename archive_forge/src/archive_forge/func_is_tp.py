from __future__ import annotations
import copy
import sys
from abc import abstractmethod
from numbers import Number, Integral
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _transform_rep
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
def is_tp(self, atol: float | None=None, rtol: float | None=None) -> bool:
    """Test if a channel is trace-preserving (TP)"""
    choi = _to_choi(self._channel_rep, self._data, *self.dim)
    return self._is_tp_helper(choi, atol, rtol)