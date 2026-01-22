import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('alg', (nx.pagerank, _pagerank_python))
def test_one_nonzero_personalization_value(self, alg):
    G = nx.complete_graph(4)
    personalize = {0: 0, 1: 0, 2: 0, 3: 1}
    answer = {0: 0.22077931820379187, 1: 0.22077931820379187, 2: 0.22077931820379187, 3: 0.3376620453886241}
    p = alg(G, alpha=0.85, personalization=personalize)
    for n in G:
        assert p[n] == pytest.approx(answer[n], abs=0.0001)