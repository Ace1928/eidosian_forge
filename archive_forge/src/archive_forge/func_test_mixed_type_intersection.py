import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_mixed_type_intersection():
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        H = nx.MultiGraph()
        I = nx.Graph()
        U = nx.intersection_all([G, H, I])
    with pytest.raises(nx.NetworkXError):
        X = nx.Graph()
        Y = nx.DiGraph()
        XY = nx.intersection_all([X, Y])