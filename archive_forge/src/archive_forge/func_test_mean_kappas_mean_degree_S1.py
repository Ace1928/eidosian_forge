import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_mean_kappas_mean_degree_S1():
    G = nx.geometric_soft_configuration_graph(beta=2.5, n=50, gamma=2.7, mean_degree=10, seed=8023)
    kappas = nx.get_node_attributes(G, 'kappa')
    mean_kappas = sum(kappas.values()) / len(kappas)
    assert math.fabs(mean_kappas - 10) < 0.5
    degrees = dict(G.degree())
    mean_degree = sum(degrees.values()) / len(degrees)
    assert math.fabs(mean_degree - 10) < 1