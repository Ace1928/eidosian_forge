import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def zip_items(self, other: 'cirq.PauliString[TKey]') -> Iterator[Tuple[TKey, Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]]:
    """Combines pauli operations from pauli strings in a qubit-by-qubit fashion.

        For every qubit that has a `cirq.Pauli` operation acting on it in both `self` and `other`,
        the method yields a tuple corresponding to `(qubit, (pauli_in_self, pauli_in_other))`.

        Args:
            other: The other `cirq.PauliString` to zip pauli operations with.

        Returns:
            A sequence of `(qubit, (pauli_in_self, pauli_in_other))` tuples for every `qubit`
            that has a `cirq.Pauli` operation acting on it in both `self` and `other.
        """
    for qubit, pauli0 in self.items():
        if qubit in other:
            yield (qubit, (pauli0, other[qubit]))