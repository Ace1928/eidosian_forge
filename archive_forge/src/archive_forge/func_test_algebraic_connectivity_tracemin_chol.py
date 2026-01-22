from math import sqrt
import pytest
import networkx as nx
def test_algebraic_connectivity_tracemin_chol():
    """Test that "tracemin_chol" raises an exception."""
    pytest.importorskip('scipy')
    G = nx.barbell_graph(5, 4)
    with pytest.raises(nx.NetworkXError):
        nx.algebraic_connectivity(G, method='tracemin_chol')