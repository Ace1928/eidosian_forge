import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_compare_mean_kappas_different_gammas_S1():
    G1 = nx.geometric_soft_configuration_graph(beta=1.5, n=20, gamma=2.7, mean_degree=5, seed=42)
    G2 = nx.geometric_soft_configuration_graph(beta=1.5, n=20, gamma=3.5, mean_degree=5, seed=42)
    kappas1 = nx.get_node_attributes(G1, 'kappa')
    mean_kappas1 = sum(kappas1.values()) / len(kappas1)
    kappas2 = nx.get_node_attributes(G2, 'kappa')
    mean_kappas2 = sum(kappas2.values()) / len(kappas2)
    assert math.fabs(mean_kappas1 - mean_kappas2) < 1