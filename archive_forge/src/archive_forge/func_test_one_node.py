from itertools import chain, combinations
import pytest
import networkx as nx
def test_one_node():
    test = nx.Graph()
    test.add_node('a')
    ground_truth = {frozenset(['a'])}
    communities = nx.community.label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth