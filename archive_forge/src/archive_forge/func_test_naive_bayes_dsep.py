from itertools import combinations
import pytest
import networkx as nx
def test_naive_bayes_dsep(naive_bayes_graph):
    """Example-based test of d-separation for naive_bayes_graph."""
    for u, v in combinations(range(1, 5), 2):
        assert nx.d_separated(naive_bayes_graph, {u}, {v}, {0})
        assert not nx.d_separated(naive_bayes_graph, {u}, {v}, {})