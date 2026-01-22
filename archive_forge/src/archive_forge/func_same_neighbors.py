import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def same_neighbors(u, v):
    return u not in G[v] and v not in G[u] and (G[u] == G[v])