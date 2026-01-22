import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_monomorphism_iter1():
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g1.add_edge('A', 'B')
    g1.add_edge('B', 'C')
    g1.add_edge('C', 'A')
    g2.add_edge('X', 'Y')
    g2.add_edge('Y', 'Z')
    gm12 = iso.DiGraphMatcher(g1, g2)
    x = list(gm12.subgraph_monomorphisms_iter())
    assert {'A': 'X', 'B': 'Y', 'C': 'Z'} in x
    assert {'A': 'Y', 'B': 'Z', 'C': 'X'} in x
    assert {'A': 'Z', 'B': 'X', 'C': 'Y'} in x
    assert len(x) == 3
    gm21 = iso.DiGraphMatcher(g2, g1)
    assert not gm21.subgraph_is_monomorphic()