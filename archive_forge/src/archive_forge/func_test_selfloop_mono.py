import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_selfloop_mono():
    edges0 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 1), (3, 2), (4, 2), (4, 5), (5, 4)]
    edges = edges0 + [(2, 2)]
    nodes = list(range(6))
    for g1 in [nx.Graph(), nx.DiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(100):
            new_nodes = list(nodes)
            random.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            g2.remove_edges_from(nx.selfloop_edges(g2))
            if not g1.is_directed():
                gm = iso.GraphMatcher(g2, g1)
            else:
                gm = iso.DiGraphMatcher(g2, g1)
            assert not gm.subgraph_is_monomorphic()