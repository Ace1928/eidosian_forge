import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
@pytest.mark.filterwarnings('ignore')
def test_random_tree_using_generator():
    """Tests that creating a random tree with a generator works"""
    G = nx.Graph()
    T = nx.random_tree(10, seed=1234, create_using=G)
    assert nx.is_tree(T)