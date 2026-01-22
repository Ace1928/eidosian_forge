import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_from_numpy_array_dtype(self):
    dt = [('weight', float), ('cost', int)]
    A = np.array([[(1.0, 2)]], dtype=dt)
    G = nx.from_numpy_array(A)
    assert type(G[0][0]['weight']) == float
    assert type(G[0][0]['cost']) == int
    assert G[0][0]['cost'] == 2
    assert G[0][0]['weight'] == 1.0