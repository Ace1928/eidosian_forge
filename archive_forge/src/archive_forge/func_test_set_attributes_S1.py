import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_set_attributes_S1():
    G = nx.geometric_soft_configuration_graph(beta=1.5, n=100, gamma=2.7, mean_degree=10, seed=42)
    kappas = nx.get_node_attributes(G, 'kappa')
    assert len(kappas) == 100
    thetas = nx.get_node_attributes(G, 'theta')
    assert len(thetas) == 100
    radii = nx.get_node_attributes(G, 'radius')
    assert len(radii) == 100