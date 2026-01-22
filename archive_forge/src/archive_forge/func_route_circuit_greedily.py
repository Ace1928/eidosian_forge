import itertools
from typing import (
import numpy as np
import networkx as nx
from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph
def route_circuit_greedily(circuit: circuits.Circuit, device_graph: nx.Graph, **kwargs) -> SwapNetwork:
    """Greedily routes a circuit on a given device.

    Alternates between heuristically picking a few SWAPs to change the mapping
    and applying all logical operations possible given the new mapping, until
    all logical operations have been applied.

    The SWAP selection heuristic is as follows. In every iteration, the
    remaining two-qubit gates are partitioned into time slices. (See
    utils.get_time_slices for details.) For each set of candidate SWAPs, the new
    mapping is computed. For each time slice and every two-qubit gate therein,
    the distance of the two logical qubits in the device graph under the new
    mapping is calculated. A candidate set 'S' of SWAPs is taken out of
    consideration if for some other set 'T' there is a time slice such that all
    of the distances for 'T' are at most those for 'S' (and they are not all
    equal).

    If more than one candidate remains, the size of the set of SWAPs considered
    is increased by one and the process is repeated. If after considering SWAP
    sets of size up to 'max_search_radius', more than one candidate remains,
    then the pairs of qubits in the first time slice are considered, and those
    farthest away under the current mapping are brought together using SWAPs
    using a shortest path in the device graph.

    Args:
        circuit: The circuit to route.
        device_graph: The device's graph, in which each vertex is a qubit
            and each edge indicates the ability to do an operation on those
            qubits.
        **kwargs: Further keyword args, including
            max_search_radius: The maximum number of disjoint device edges to
                consider routing on.
            max_num_empty_steps: The maximum number of swap sets to apply
                without allowing a new logical operation to be performed.
            initial_mapping: The initial mapping of physical to logical qubits
                to use. Defaults to a greedy initialization.
            can_reorder: A predicate that determines if two operations may be
                reordered.
            random_state: Random state or random state seed.
    """
    router = _GreedyRouter(circuit, device_graph, **kwargs)
    router.route()
    swap_network = router.swap_network
    swap_network.circuit = circuits.Circuit(swap_network.circuit.all_operations())
    return swap_network