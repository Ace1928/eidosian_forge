import collections
from typing import cast, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from cirq import circuits, ops, transformers
from cirq.contrib.acquaintance.gates import SwapNetworkGate, AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.devices import get_acquaintance_size
from cirq.contrib.acquaintance.permutation import PermutationGate
def replace_acquaintance_with_swap_network(circuit: 'cirq.Circuit', qubit_order: Sequence['cirq.Qid'], acquaintance_size: Optional[int]=0, swap_gate: 'cirq.Gate'=ops.SWAP) -> bool:
    """Replace every rectified moment with acquaintance gates with a generalized swap network.

    The generalized swap network has a partition given by the acquaintance gates in that moment
    (and singletons for the free qubits). Accounts for reversing effect of swap networks.

    Args:
        circuit: The acquaintance strategy.
        qubit_order: The qubits, in order, on which the replacing swap network
            gate acts on.
        acquaintance_size: The acquaintance size of the new swap network gate.
        swap_gate: The gate used to swap logical indices.

    Returns: Whether or not the overall effect of the inserted swap network
        gates is to reverse the order of the qubits, i.e. the parity of the
        number of swap network gates inserted.

    Raises:
        TypeError: circuit is not an acquaintance strategy.
    """
    rectify_acquaintance_strategy(circuit)
    reflected = False
    reverse_map = {q: r for q, r in zip(qubit_order, reversed(qubit_order))}
    for moment_index, moment in enumerate(circuit):
        if reflected:
            moment = moment.transform_qubits(reverse_map.__getitem__)
        if all((isinstance(op.gate, AcquaintanceOpportunityGate) for op in moment.operations)):
            swap_network_gate = SwapNetworkGate.from_operations(qubit_order, moment.operations, acquaintance_size, swap_gate)
            swap_network_op = swap_network_gate(*qubit_order)
            moment = circuits.Moment([swap_network_op])
            reflected = not reflected
        circuit._moments[moment_index] = moment
    return reflected