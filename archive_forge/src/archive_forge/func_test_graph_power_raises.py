import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_graph_power_raises():
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.power(nx.MultiDiGraph(), 2)