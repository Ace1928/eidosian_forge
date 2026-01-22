import itertools
import pytest
import networkx as nx
def max_degree(G):
    """Get the maximum degree of any node in G."""
    return max((G.degree(node) for node in G.nodes)) if len(G.nodes) > 0 else 0