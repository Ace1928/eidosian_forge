import pytest
import networkx as nx
def test_scale_and_center_arg(self):
    sc = self.check_scale_and_center
    c = (4, 5)
    G = nx.complete_graph(9)
    G.add_node(9)
    sc(nx.random_layout(G, center=c), scale=0.5, center=(4.5, 5.5))
    sc(nx.spring_layout(G, scale=2, center=c), scale=2, center=c)
    sc(nx.spectral_layout(G, scale=2, center=c), scale=2, center=c)
    sc(nx.circular_layout(G, scale=2, center=c), scale=2, center=c)
    sc(nx.shell_layout(G, scale=2, center=c), scale=2, center=c)
    sc(nx.spiral_layout(G, scale=2, center=c), scale=2, center=c)
    sc(nx.kamada_kawai_layout(G, scale=2, center=c), scale=2, center=c)
    c = (2, 3, 5)
    sc(nx.kamada_kawai_layout(G, dim=3, scale=2, center=c), scale=2, center=c)