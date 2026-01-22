import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def get_grid_interaction_layer_circuit(device_graph: nx.Graph, pattern: Sequence[GridInteractionLayer]=HALF_GRID_STAGGERED_PATTERN, two_qubit_gate=ops.ISWAP ** 0.5) -> 'cirq.Circuit':
    """Create a circuit representation of a grid interaction pattern on a given device topology.

    The resulting circuit is deterministic, of depth len(pattern), and consists of `two_qubit_gate`
    applied to each pair in `pattern` restricted to available connections in `device_graph`.

    Args:
        device_graph: A graph whose nodes are qubits and whose edges represent the possibility of
            doing a two-qubit gate. This combined with the `pattern` argument determines which
            two qubit pairs are activated when.
        pattern: A sequence of `GridInteractionLayer`, each of which has a particular set of
            qubits that are activated simultaneously. These pairs of qubits are deduced by
            combining this argument with `device_graph`.
        two_qubit_gate: The two qubit gate to use in constructing the circuit layers.
    """
    moments = []
    for layer in pattern:
        pairs = sorted(_get_active_pairs(device_graph, layer))
        if len(pairs) == 0:
            continue
        moments += [circuits.Moment((two_qubit_gate.on(*pair) for pair in pairs))]
    return circuits.Circuit(moments)