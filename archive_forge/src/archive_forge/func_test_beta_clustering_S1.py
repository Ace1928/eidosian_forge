import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_beta_clustering_S1():
    G1 = nx.geometric_soft_configuration_graph(beta=1.5, n=100, gamma=3.5, mean_degree=10, seed=42)
    G2 = nx.geometric_soft_configuration_graph(beta=3.0, n=100, gamma=3.5, mean_degree=10, seed=42)
    assert nx.average_clustering(G1) < nx.average_clustering(G2)