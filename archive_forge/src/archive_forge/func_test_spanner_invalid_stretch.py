import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_invalid_stretch():
    """Check whether an invalid stretch is caught."""
    with pytest.raises(ValueError):
        G = nx.empty_graph()
        nx.spanner(G, 0)