from functools import partial
import pytest
import networkx as nx
def test_bfs_edges_sorting(self):
    D = nx.DiGraph()
    D.add_edges_from([(0, 1), (0, 2), (1, 4), (1, 3), (2, 5)])
    sort_desc = partial(sorted, reverse=True)
    edges_asc = nx.bfs_edges(D, source=0, sort_neighbors=sorted)
    edges_desc = nx.bfs_edges(D, source=0, sort_neighbors=sort_desc)
    assert list(edges_asc) == [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    assert list(edges_desc) == [(0, 2), (0, 1), (2, 5), (1, 4), (1, 3)]