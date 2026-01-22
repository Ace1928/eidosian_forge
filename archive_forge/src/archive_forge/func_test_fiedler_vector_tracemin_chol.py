from math import sqrt
import pytest
import networkx as nx
def test_fiedler_vector_tracemin_chol():
    """Test that "tracemin_chol" raises an exception."""
    pytest.importorskip('scipy')
    G = nx.barbell_graph(5, 4)
    with pytest.raises(nx.NetworkXError):
        nx.fiedler_vector(G, method='tracemin_chol')