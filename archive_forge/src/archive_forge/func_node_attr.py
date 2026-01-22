import pytest
import networkx as nx
def node_attr(u):
    return G.nodes[u].get('size', 0.5) * 3