from itertools import combinations
import pytest
import networkx as nx
def test_asia_graph_dsep(asia_graph):
    """Example-based test of d-separation for asia_graph."""
    assert nx.d_separated(asia_graph, {'asia', 'smoking'}, {'dyspnea', 'xray'}, {'bronchitis', 'either'})
    assert nx.d_separated(asia_graph, {'tuberculosis', 'cancer'}, {'bronchitis'}, {'smoking', 'xray'})