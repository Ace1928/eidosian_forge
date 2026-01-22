from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def decompose_gate(self) -> Sequence['cirq.Gate']:
    """Decomposes this clifford into a series of H and pauli rotation gates.

        Returns:
            A sequence of H and pauli rotation gates which are equivalent to this
            clifford gate if applied in order. This decomposition agrees with
            cirq.unitary(self), including global phase.
        """
    if self == SingleQubitCliffordGate.H:
        return [common_gates.H]
    rotations = self.decompose_rotation()
    return [r ** (qt / 2) for r, qt in rotations]