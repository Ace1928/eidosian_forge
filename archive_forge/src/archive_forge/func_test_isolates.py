import networkx as nx
from networkx.utils import pairwise
def test_isolates(self):
    """Tests that a graph with isolated nodes has all isolates in
        one block of the partition.

        """
    G = nx.empty_graph(5)
    cells = nx.voronoi_cells(G, {0, 2, 4})
    expected = {0: {0}, 2: {2}, 4: {4}, 'unreachable': {1, 3}}
    assert expected == cells