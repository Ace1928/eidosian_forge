import itertools
import networkx as nx
def test_visibility_graph__empty_series__empty_graph():
    null_graph = nx.visibility_graph([])
    assert nx.is_empty(null_graph)