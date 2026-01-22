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
def inplace_before(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
    """Propagates the pauli string from after to before a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C^\\dagger P C$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.
        """
    return self.inplace_after(protocols.inverse(ops))