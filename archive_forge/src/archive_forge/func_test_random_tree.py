import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
@pytest.mark.filterwarnings('ignore')
def test_random_tree():
    """Tests that a random tree is in fact a tree."""
    T = nx.random_tree(10, seed=1234)
    assert nx.is_tree(T)