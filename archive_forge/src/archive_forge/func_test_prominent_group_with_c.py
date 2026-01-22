import pytest
import networkx as nx
def test_prominent_group_with_c(self):
    """
        Prominent group without some nodes
        """
    G = nx.path_graph(5)
    k = 1
    b, g = nx.prominent_group(G, k, normalized=False, C=[2])
    b_answer, g_answer = (3.0, [1])
    assert b == b_answer and g == g_answer