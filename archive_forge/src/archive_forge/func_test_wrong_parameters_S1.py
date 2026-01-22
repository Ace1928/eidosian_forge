import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_wrong_parameters_S1():
    with pytest.raises(nx.NetworkXError, match='Please provide either kappas, or all 3 of: n, gamma and mean_degree.'):
        G = nx.geometric_soft_configuration_graph(beta=1.5, gamma=3.5, mean_degree=10, seed=42)
    with pytest.raises(nx.NetworkXError, match='When kappas is input, n, gamma and mean_degree must not be.'):
        kappas = {i: 10 for i in range(1000)}
        G = nx.geometric_soft_configuration_graph(beta=1.5, kappas=kappas, gamma=2.3, seed=42)
    with pytest.raises(nx.NetworkXError, match='Please provide either kappas, or all 3 of: n, gamma and mean_degree.'):
        G = nx.geometric_soft_configuration_graph(beta=1.5, seed=42)