import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_multigraph_nonnumeric_key(self):
    for MG in (nx.MultiGraph, nx.MultiDiGraph):
        for cc in (True, False):
            G = nx.MultiGraph()
            G.add_edge(0, 1, key='I', value='a')
            G.add_edge(0, 2, key='II', value='b')
            G.add_edge(0, 3, key='II', value='c')
            mapping = {1: 4, 2: 4, 3: 4}
            nx.relabel_nodes(G, mapping, copy=False)
            assert {'value': 'a'} in G[0][4].values()
            assert {'value': 'b'} in G[0][4].values()
            assert {'value': 'c'} in G[0][4].values()
            assert 0 in G[0][4]
            assert 'I' in G[0][4]
            assert 'II' in G[0][4]