from typing import Any, Dict, List, Sequence, TYPE_CHECKING, Tuple
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import gate_features, common_gates, eigen_gate, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
Inits PauliInteractionGate.

        Args:
            pauli0: The interaction axis for the first qubit.
            invert0: Whether to condition on the +1 or -1 eigenvector of the
                first qubit's interaction axis.
            pauli1: The interaction axis for the second qubit.
            invert1: Whether to condition on the +1 or -1 eigenvector of the
                second qubit's interaction axis.
            exponent: Determines the amount of phasing to apply to the vector
                equal to the tensor product of the two conditions.
        