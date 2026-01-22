from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def dense_pauli_string(self, pauli: Pauli) -> 'cirq.DensePauliString':
    from cirq.ops import dense_pauli_string
    pauli_tuple = self.pauli_tuple(pauli)
    coefficient = -1 if pauli_tuple[1] else 1
    return dense_pauli_string.DensePauliString(str(pauli_tuple[0]), coefficient=coefficient)