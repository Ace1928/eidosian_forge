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
def get_distance_vector(self, logical_edges: Iterable[QidPair], swaps: Sequence[QidPair]):
    """Gets distances between physical qubits mapped to by given logical
        edges, after specified SWAPs are applied."""
    self.update_mapping(*swaps)
    distance_vector = np.array([self.distance(e) for e in logical_edges])
    self.update_mapping(*swaps)
    return distance_vector