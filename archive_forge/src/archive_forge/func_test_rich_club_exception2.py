import pytest
import networkx as nx
def test_rich_club_exception2():
    with pytest.raises(nx.NetworkXNotImplemented):
        G = nx.MultiGraph()
        nx.rich_club_coefficient(G)