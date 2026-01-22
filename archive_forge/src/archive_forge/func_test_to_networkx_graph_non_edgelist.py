import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_networkx_graph_non_edgelist():
    invalid_edgelist = [1, 2, 3]
    with pytest.raises(nx.NetworkXError, match='Input is not a valid edge list'):
        nx.to_networkx_graph(invalid_edgelist)