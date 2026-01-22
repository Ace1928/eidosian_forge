import pytest
import networkx as nx
def test_random_kernel_graph(self):

    def integral(u, w, z):
        return c * (z - w)

    def root(u, w, r):
        return r / c + w
    c = 1
    graph = nx.random_kernel_graph(1000, integral, root)
    graph = nx.random_kernel_graph(1000, integral, root, seed=42)
    assert len(graph) == 1000