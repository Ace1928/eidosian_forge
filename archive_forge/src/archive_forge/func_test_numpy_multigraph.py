import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize(('operator', 'expected'), ((sum, 77), (min, 7), (max, 70)))
def test_numpy_multigraph(multigraph_test_graph, operator, expected):
    A = nx.to_numpy_array(multigraph_test_graph, multigraph_weight=operator)
    assert A[1, 0] == expected