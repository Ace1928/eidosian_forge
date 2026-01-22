import pytest
import networkx as nx
def test_richclub3():
    G = nx.karate_club_graph()
    rc = nx.rich_club_coefficient(G, normalized=False)
    assert rc == {0: 156.0 / 1122, 1: 154.0 / 1056, 2: 110.0 / 462, 3: 78.0 / 240, 4: 44.0 / 90, 5: 22.0 / 42, 6: 10.0 / 20, 7: 10.0 / 20, 8: 10.0 / 20, 9: 6.0 / 12, 10: 2.0 / 6, 11: 2.0 / 6, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0}