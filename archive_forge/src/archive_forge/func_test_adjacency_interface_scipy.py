import pytest
import networkx as nx
def test_adjacency_interface_scipy(self):
    A = nx.to_scipy_sparse_array(self.Gs, dtype='d')
    pos = nx.drawing.layout._sparse_fruchterman_reingold(A)
    assert pos.shape == (6, 2)
    pos = nx.drawing.layout._sparse_spectral(A)
    assert pos.shape == (6, 2)
    pos = nx.drawing.layout._sparse_fruchterman_reingold(A, dim=3)
    assert pos.shape == (6, 3)