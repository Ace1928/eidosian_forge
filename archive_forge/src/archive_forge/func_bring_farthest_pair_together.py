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
def bring_farthest_pair_together(self, pairs: Sequence[QidPair]):
    """Adds SWAPs to bring the farthest-apart pair of logical qubits
        together."""
    distances = [self.distance(pair) for pair in pairs]
    assert distances
    max_distance = min(distances)
    farthest_pairs = [pair for pair, d in zip(pairs, distances) if d == max_distance]
    choice = self.prng.choice(len(farthest_pairs))
    farthest_pair = farthest_pairs[choice]
    edge = self.log_to_phys(*farthest_pair)
    shortest_path = nx.shortest_path(self.device_graph, *edge)
    assert len(shortest_path) - 1 == max_distance
    midpoint = max_distance // 2
    self.swap_along_path(shortest_path[:midpoint])
    self.swap_along_path(shortest_path[midpoint:])