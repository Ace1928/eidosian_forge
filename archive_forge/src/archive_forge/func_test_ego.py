import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_ego(self):
    G = nx.star_graph(3)
    H = nx.ego_graph(G, 0)
    assert nx.is_isomorphic(G, H)
    G.add_edge(1, 11)
    G.add_edge(2, 22)
    G.add_edge(3, 33)
    H = nx.ego_graph(G, 0)
    assert nx.is_isomorphic(nx.star_graph(3), H)
    G = nx.path_graph(3)
    H = nx.ego_graph(G, 0)
    assert edges_equal(H.edges(), [(0, 1)])
    H = nx.ego_graph(G, 0, undirected=True)
    assert edges_equal(H.edges(), [(0, 1)])
    H = nx.ego_graph(G, 0, center=False)
    assert edges_equal(H.edges(), [])