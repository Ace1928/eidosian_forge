import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_is_k_regular3(self):
    g = gen.lollipop_graph(5, 5)
    assert not reg.is_k_regular(g, 5)
    assert not reg.is_k_regular(g, 6)