from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def equivalent_gate_before(self, after: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
    """Returns a SingleQubitCliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
    return self.merged_with(after).merged_with(self ** (-1))