import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_corona_product():
    G = nx.cycle_graph(3)
    H = nx.path_graph(2)
    C = nx.corona_product(G, H)
    assert len(C) == len(G) * len(H) + len(G)
    assert C.size() == G.size() + len(G) * H.size() + len(G) * len(H)