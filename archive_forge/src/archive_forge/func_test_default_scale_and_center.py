import pytest
import networkx as nx
def test_default_scale_and_center(self):
    sc = self.check_scale_and_center
    c = (0, 0)
    G = nx.complete_graph(9)
    G.add_node(9)
    sc(nx.random_layout(G), scale=0.5, center=(0.5, 0.5))
    sc(nx.spring_layout(G), scale=1, center=c)
    sc(nx.spectral_layout(G), scale=1, center=c)
    sc(nx.circular_layout(G), scale=1, center=c)
    sc(nx.shell_layout(G), scale=1, center=c)
    sc(nx.spiral_layout(G), scale=1, center=c)
    sc(nx.kamada_kawai_layout(G), scale=1, center=c)
    c = (0, 0, 0)
    sc(nx.kamada_kawai_layout(G, dim=3), scale=1, center=c)