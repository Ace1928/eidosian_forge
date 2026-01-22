import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def get_random_combinations_for_device(n_library_circuits: int, n_combinations: int, device_graph: nx.Graph, *, pattern: Sequence[GridInteractionLayer]=HALF_GRID_STAGGERED_PATTERN, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[CircuitLibraryCombination]:
    """For a given device, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        device_graph: A graph whose nodes are qubits and whose edges represent
            the possibility of doing a two-qubit gate. This combined with the
            `pattern` argument determines which two qubit pairs are activated
            when.
        pattern: A sequence of `GridInteractionLayer`, each of which has
            a particular set of qubits that are activated simultaneously. These
            pairs of qubits are deduced by combining this argument with `device_graph`.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to an interaction
        layer in `pattern` where there is a non-zero number of pairs which would be activated.
        Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))` where `len(pairs)` may
        be different for each entry (i.e. for each layer in `pattern`). This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """

    def pair_gen():
        for layer in pattern:
            pairs = sorted(_get_active_pairs(device_graph, layer))
            if len(pairs) == 0:
                continue
            yield (pairs, layer)
    return _get_random_combinations(n_library_circuits=n_library_circuits, n_combinations=n_combinations, random_state=random_state, pair_gen=pair_gen())