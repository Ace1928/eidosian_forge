import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_geometric_edges_raises_no_pos():
    G = nx.path_graph(3)
    msg = "all nodes. must have a '"
    with pytest.raises(nx.NetworkXError, match=msg):
        nx.geometric_edges(G, radius=1)