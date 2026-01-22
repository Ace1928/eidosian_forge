import itertools
from typing import cast, Dict, Hashable, TYPE_CHECKING
import networkx as nx
from sortedcontainers import SortedDict, SortedSet
from cirq import ops, value
Gets an initial mapping of logical to physical qubits for routing.

    Args:
        logical_graph: The graph whose edges correspond to pairs of qubits that
            should be mapped to nearby physical qubits.
        device_graph: The graph of the device.
        random_state: Random state or random state seed.

    The mapping starts by mapping the center of the logical graph to the center
    of the physical graph. Subsequent logical qubits are mapped to physical
    qubits greedily. At each iteration, the logical qubits with the largest
    number of already mapped neighbors and the physical qubits neighboring
    those already mapped to are considered. The pair of logical and physical
    qubits that minimizes the average distance to already mapped logical
    neighbors is selected.
    