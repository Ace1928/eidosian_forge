from typing import Dict, List, Tuple, TYPE_CHECKING
import cirq
def chip_as_adjacency_list(device: 'cirq_google.GridDevice') -> Dict[cirq.GridQubit, List[cirq.GridQubit]]:
    """Gives adjacency list representation of a chip.

    The adjacency list is constructed in order of above, left_of, below and
    right_of consecutively.

    Args:
        device: Chip to be converted.

    Returns:
        Map from nodes to list of qubits which represent all the neighbours of
        given qubit.
    """
    c_set = device.metadata.qubit_set
    c_adj: Dict[cirq.GridQubit, List[cirq.GridQubit]] = {}
    for n in c_set:
        c_adj[n] = []
        for m in [above(n), left_of(n), below(n), right_of(n)]:
            if m in c_set:
                c_adj[n].append(m)
    return c_adj