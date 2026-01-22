import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_monomorphism_edge_match():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, label='A')
    G.add_edge(2, 1, label='B')
    G.add_edge(2, 2, label='C')
    SG = nx.DiGraph()
    SG.add_node(5)
    SG.add_node(6)
    SG.add_edge(5, 6, label='A')
    gm = iso.DiGraphMatcher(G, SG, edge_match=iso.categorical_edge_match('label', None))
    assert gm.subgraph_is_monomorphic()