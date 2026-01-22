import itertools
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.contrib.acquaintance.permutation import SwapPermutationGate, PermutationGate
Construct a circular shift gate.

        Args:
            num_qubits: The number of qubits to shift.
            shift: The number of positions to circularly left shift the qubits.
            swap_gate: The gate to use when decomposing.
        