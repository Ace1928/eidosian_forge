from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@classmethod
def from_unitary_with_global_phase(cls, u: np.ndarray) -> Optional[Tuple['SingleQubitCliffordGate', complex]]:
    """Creates Clifford gate with given unitary, including global phase.

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            A tuple of a SingleQubitCliffordGate and a global phase, such that
            the gate unitary (as given by `cirq.unitary`) times the global phase
            is identical to the given unitary `u`; or `None` if `u` is not the
            matrix of a single-qubit Clifford gate.
        """
    gate = cls.from_unitary(u)
    if gate is None:
        return None
    k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
    return (gate, u[k] / protocols.unitary(gate)[k])