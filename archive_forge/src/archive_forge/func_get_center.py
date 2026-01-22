import itertools
from typing import cast, Dict, Hashable, TYPE_CHECKING
import networkx as nx
from sortedcontainers import SortedDict, SortedSet
from cirq import ops, value
def get_center(graph: nx.Graph) -> Hashable:
    centralities = nx.betweenness_centrality(graph)
    return max(centralities, key=centralities.get)