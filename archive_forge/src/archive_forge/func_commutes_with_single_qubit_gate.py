from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def commutes_with_single_qubit_gate(self, gate: 'SingleQubitCliffordGate') -> bool:
    """Tests if the two circuits would be equivalent up to global phase:
        --self--gate-- and --gate--self--"""
    self_then_gate = self.clifford_tableau.then(gate.clifford_tableau)
    gate_then_self = gate.clifford_tableau.then(self.clifford_tableau)
    return self_then_gate == gate_then_self