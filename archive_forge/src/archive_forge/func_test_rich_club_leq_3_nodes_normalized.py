import pytest
import networkx as nx
def test_rich_club_leq_3_nodes_normalized():
    G = nx.Graph()
    with pytest.raises(nx.exception.NetworkXError, match='Graph has fewer than four nodes'):
        rc = nx.rich_club_coefficient(G, normalized=True)
    for i in range(3):
        G.add_node(i)
        with pytest.raises(nx.exception.NetworkXError, match='Graph has fewer than four nodes'):
            rc = nx.rich_club_coefficient(G, normalized=True)