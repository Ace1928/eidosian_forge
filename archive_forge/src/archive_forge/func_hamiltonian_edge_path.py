import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def hamiltonian_edge_path(G, source):
    source = arbitrary_element(G)
    neighbors = set(G[source]) - {source}
    n = len(G)
    for target in neighbors:
        for path in nx.all_simple_edge_paths(G, source, target):
            if len(path) == n - 1:
                yield path