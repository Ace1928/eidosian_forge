import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
@pytest.mark.parametrize('method', ('kou', 'mehlhorn'))
def test_steiner_tree_multigraph_weight_attribute(method):
    G = nx.cycle_graph(3, create_using=nx.MultiGraph)
    nx.set_edge_attributes(G, {e: 10 for e in G.edges}, name='distance')
    G.add_edge(2, 0, distance=5)
    H = nx.approximation.steiner_tree(G, list(G), method=method, weight='distance')
    assert len(H.edges) == 2 and H.has_edge(2, 0, key=1)
    assert sum((dist for *_, dist in H.edges(data='distance'))) == 15