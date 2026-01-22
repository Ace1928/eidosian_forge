from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_pauli(pauli: Pauli, sqrt: bool=False) -> 'SingleQubitCliffordGate':
    prev_pauli = Pauli.by_relative_index(pauli, -1)
    next_pauli = Pauli.by_relative_index(pauli, 1)
    if sqrt:
        rotation_map = {prev_pauli: (next_pauli, True), pauli: (pauli, False), next_pauli: (prev_pauli, False)}
    else:
        rotation_map = {prev_pauli: (prev_pauli, True), pauli: (pauli, False), next_pauli: (next_pauli, True)}
    return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))