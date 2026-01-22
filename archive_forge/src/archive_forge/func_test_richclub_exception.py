import pytest
import networkx as nx
def test_richclub_exception():
    with pytest.raises(nx.NetworkXNotImplemented):
        G = nx.DiGraph()
        nx.rich_club_coefficient(G)