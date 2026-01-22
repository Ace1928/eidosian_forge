import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_weighted_gnp_graph():
    """Test spanner construction on an weighted gnp graph."""
    G = nx.gnp_random_graph(20, 0.4, seed=_seed)
    _assign_random_weights(G, seed=_seed)
    spanner = nx.spanner(G, 4, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 4, weight='weight')
    spanner = nx.spanner(G, 10, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 10, weight='weight')