import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('alg', (nx.pagerank, _pagerank_python, nx.google_matrix))
def test_zero_personalization_vector(self, alg):
    G = nx.complete_graph(4)
    personalize = {0: 0, 1: 0, 2: 0, 3: 0}
    pytest.raises(ZeroDivisionError, alg, G, personalization=personalize)