import networkx as nx
def test_width_none(self):
    G = nx.cycle_graph(4)
    edges = nx.bfs_beam_edges(G, 0, identity, width=None)
    assert list(edges) == [(0, 3), (0, 1), (3, 2)]