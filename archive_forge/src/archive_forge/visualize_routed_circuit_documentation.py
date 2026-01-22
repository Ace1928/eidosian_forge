from typing import Dict, Optional, Tuple, TYPE_CHECKING
from cirq import circuits, ops
Returns the same circuits with information about the permutation of qubits after each swap.

    Args:
        routed_circuit: a routed circuit that potentially has inserted swaps tagged with a
            RoutingSwapTag.
        initial_map: the initial mapping from logical to physical qubits. If this is not specified
            then the identity mapping of the qubits in routed_circuit will be used as initial_map.

    Raises:
        ValueError: if a non-SWAP gate is tagged with a RoutingSwapTag.
    