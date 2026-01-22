import networkx as nx
from networkx.utils import edges_equal
def test_nonisomorphism(self):

    def f(x):
        return list(nx.nonisomorphic_trees(x))
    trees = f(6)
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            assert not nx.is_isomorphic(trees[i], trees[j])
    trees = f(8)
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            assert not nx.is_isomorphic(trees[i], trees[j])