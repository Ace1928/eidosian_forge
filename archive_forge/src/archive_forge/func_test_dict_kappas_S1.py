import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_dict_kappas_S1():
    kappas = {i: 10 for i in range(1000)}
    G = nx.geometric_soft_configuration_graph(beta=1, kappas=kappas)
    assert len(G) == 1000
    kappas = nx.get_node_attributes(G, 'kappa')
    assert all((kappa == 10 for kappa in kappas.values()))