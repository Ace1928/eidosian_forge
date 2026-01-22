import pytest
import networkx as nx
def test_rich_club_selfloop():
    G = nx.Graph()
    G.add_edge(1, 1)
    G.add_edge(1, 2)
    with pytest.raises(Exception, match='rich_club_coefficient is not implemented for graphs with self loops.'):
        nx.rich_club_coefficient(G)