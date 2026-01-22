import networkx as nx
from networkx.utils import edges_equal
def test_tree_structure(self):

    def f(x):
        return list(nx.nonisomorphic_trees(x))
    for i in f(6):
        assert nx.is_tree(i)
    for i in f(8):
        assert nx.is_tree(i)