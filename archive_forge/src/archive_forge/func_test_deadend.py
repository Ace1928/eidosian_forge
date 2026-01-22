import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_deadend(self):
    """Check if one-node cycles are handled properly. Taken from ticket
        #2906 from @sshraven."""
    G = nx.DiGraph()
    G.add_nodes_from(range(5), demand=0)
    G.nodes[4]['demand'] = -13
    G.nodes[3]['demand'] = 13
    G.add_edges_from([(0, 2), (0, 3), (2, 1)], capacity=20, weight=0.1)
    pytest.raises(nx.NetworkXUnfeasible, nx.min_cost_flow, G)