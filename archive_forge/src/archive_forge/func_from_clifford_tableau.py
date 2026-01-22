from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_clifford_tableau(tableau: qis.CliffordTableau) -> 'SingleQubitCliffordGate':
    if not isinstance(tableau, qis.CliffordTableau):
        raise ValueError('Input argument has to be a CliffordTableau instance.')
    if not tableau._validate():
        raise ValueError('Input tableau is not a valid Clifford tableau.')
    if tableau.n != 1:
        raise ValueError('The number of qubit of input tableau should be 1 for SingleQubitCliffordGate.')
    return SingleQubitCliffordGate(_clifford_tableau=tableau)