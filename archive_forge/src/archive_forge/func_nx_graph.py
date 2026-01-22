import abc
from typing import TYPE_CHECKING, Optional, FrozenSet, Iterable
import networkx as nx
from cirq import value
@property
def nx_graph(self) -> 'nx.Graph':
    """Returns a nx.Graph where nodes are qubits and edges are couple-able qubits.

        Returns:
            `nx.Graph` of device connectivity.
        """
    return self._nx_graph