import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
def test_rcm_alternate_heuristic():
    G = nx.Graph([(0, 0), (0, 4), (1, 1), (1, 2), (1, 5), (1, 7), (2, 2), (2, 4), (3, 3), (3, 6), (4, 4), (5, 5), (5, 7), (6, 6), (7, 7)])
    answers = [[6, 3, 5, 7, 1, 2, 4, 0], [6, 3, 7, 5, 1, 2, 4, 0], [7, 5, 1, 2, 4, 0, 6, 3]]

    def smallest_degree(G):
        deg, node = min(((d, n) for n, d in G.degree()))
        return node
    rcm = list(reverse_cuthill_mckee_ordering(G, heuristic=smallest_degree))
    assert rcm in answers