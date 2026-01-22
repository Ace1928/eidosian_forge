import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
@property
def pauli_mask(self) -> np.ndarray:
    """A 1-dimensional uint8 numpy array giving a specification of Pauli gates to use."""
    return self._pauli_mask