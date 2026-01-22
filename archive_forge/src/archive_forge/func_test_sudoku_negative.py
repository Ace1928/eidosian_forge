import pytest
import networkx as nx
def test_sudoku_negative():
    """Raise an error when generating a Sudoku graph of order -1."""
    pytest.raises(nx.NetworkXError, nx.sudoku_graph, n=-1)