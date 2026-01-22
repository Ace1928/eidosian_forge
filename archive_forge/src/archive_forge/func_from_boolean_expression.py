from collections import defaultdict
from typing import (
import numbers
import numpy as np
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from scipy.sparse import csr_matrix
from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms
@classmethod
def from_boolean_expression(cls, boolean_expr: Expr, qubit_map: Dict[str, 'cirq.Qid']) -> 'PauliSum':
    """Builds the Hamiltonian representation of a Boolean expression.

        This is based on "On the representation of Boolean and real functions as Hamiltonians for
        quantum computing" by Stuart Hadfield, https://arxiv.org/abs/1804.09130

        Args:
            boolean_expr: A Sympy expression containing symbols and Boolean operations
            qubit_map: map of string (boolean variable name) to qubit.

        Return:
            The PauliSum that represents the Boolean expression.

        Raises:
            ValueError: If `boolean_expr` is of an unsupported type.
        """
    if isinstance(boolean_expr, Symbol):
        return cls.from_pauli_strings([PauliString({}, 0.5), PauliString({qubit_map[boolean_expr.name]: pauli_gates.Z}, -0.5)])
    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_pauli_sums = [cls.from_boolean_expression(sub_boolean_expr, qubit_map) for sub_boolean_expr in boolean_expr.args]
        if isinstance(boolean_expr, And):
            pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0))
            for sub_pauli_sum in sub_pauli_sums:
                pauli_sum = pauli_sum * sub_pauli_sum
        elif isinstance(boolean_expr, Not):
            assert len(sub_pauli_sums) == 1
            pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0)) - sub_pauli_sums[0]
        elif isinstance(boolean_expr, Or):
            pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
            for sub_pauli_sum in sub_pauli_sums:
                pauli_sum = pauli_sum + sub_pauli_sum - pauli_sum * sub_pauli_sum
        elif isinstance(boolean_expr, Xor):
            pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
            for sub_pauli_sum in sub_pauli_sums:
                pauli_sum = pauli_sum + sub_pauli_sum - 2.0 * pauli_sum * sub_pauli_sum
        return pauli_sum
    raise ValueError(f'Unsupported type: {type(boolean_expr)}')