from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def pauli_tuple(self, pauli: Pauli) -> Tuple[Pauli, bool]:
    """Returns a tuple of a Pauli operator and a boolean.

        The pauli is the operator of the transform and the boolean
        determines whether the operator should be flipped.  For instance,
        it is True if the coefficient is -1, and False if the coefficient
        is 1.
        """
    x_to = self._clifford_tableau.destabilizers()[0]
    z_to = self._clifford_tableau.stabilizers()[0]
    if pauli == pauli_gates.X:
        to = x_to
    elif pauli == pauli_gates.Z:
        to = z_to
    else:
        to = x_to * z_to
        to._coefficient *= 1j
    to_gate = Pauli._XYZ[to.pauli_mask[0] - 1]
    return (to_gate, bool(to.coefficient != 1.0))