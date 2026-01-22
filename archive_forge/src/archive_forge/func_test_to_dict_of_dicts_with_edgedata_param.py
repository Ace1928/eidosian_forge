import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize('edgelist', ([(0, 1), (1, 2)], [(0, 1, {'weight': 1.0}), (1, 2, {'weight': 2.0})]))
def test_to_dict_of_dicts_with_edgedata_param(edgelist):
    G = nx.Graph()
    G.add_edges_from(edgelist)
    expected = {0: {1: 10}, 1: {0: 10, 2: 10}, 2: {1: 10}}
    assert nx.to_dict_of_dicts(G, edge_data=10) == expected