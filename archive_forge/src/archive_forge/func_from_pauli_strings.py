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
def from_pauli_strings(cls, terms: Union[PauliString, List[PauliString]]) -> 'PauliSum':
    """Returns a PauliSum by combining `cirq.PauliString` terms.

        Args:
            terms: `cirq.PauliString` or List of `cirq.PauliString`s to use inside
                of this PauliSum object.
        Returns:
            PauliSum object representing the addition of all the `cirq.PauliString`
                terms in `terms`.
        """
    if isinstance(terms, PauliString):
        terms = [terms]
    termdict: DefaultDict[UnitPauliStringT, value.Scalar] = defaultdict(lambda: 0)
    for pstring in terms:
        key = frozenset(pstring._qubit_pauli_map.items())
        termdict[key] += pstring.coefficient
    return cls(linear_dict=value.LinearDict(termdict))