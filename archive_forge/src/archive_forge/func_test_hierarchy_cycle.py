import pytest
import networkx as nx
def test_hierarchy_cycle():
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    assert nx.flow_hierarchy(G) == 0.0