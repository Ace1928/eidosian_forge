import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_mixed_type_disjoint_union():
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        H = nx.MultiGraph()
        I = nx.Graph()
        U = nx.disjoint_union_all([G, H, I])
    with pytest.raises(nx.NetworkXError):
        X = nx.Graph()
        Y = nx.DiGraph()
        XY = nx.disjoint_union_all([X, Y])