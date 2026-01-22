import pytest
import networkx as nx
def test_non_integer_input():
    pytest.raises(nx.NetworkXException, nx.is_graphical, [72.5], 'eg')
    pytest.raises(nx.NetworkXException, nx.is_graphical, [72.5], 'hh')