import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_get_edge_attributes():
    graphs = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
    for G in graphs:
        G = nx.path_graph(3, create_using=G)
        attr = 'hello'
        vals = 100
        nx.set_edge_attributes(G, vals, attr)
        attrs = nx.get_edge_attributes(G, attr)
        assert len(attrs) == 2
        for edge in G.edges:
            assert attrs[edge] == vals
        default_val = vals
        G.add_edge(4, 5)
        deafult_attrs = nx.get_edge_attributes(G, attr, default=default_val)
        assert len(deafult_attrs) == 3
        for edge in G.edges:
            assert deafult_attrs[edge] == vals