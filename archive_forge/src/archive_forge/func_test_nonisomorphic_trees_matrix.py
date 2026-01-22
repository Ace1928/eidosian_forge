import networkx as nx
from networkx.utils import edges_equal
def test_nonisomorphic_trees_matrix(self):
    trees_2 = [[[0, 1], [1, 0]]]
    assert list(nx.nonisomorphic_trees(2, create='matrix')) == trees_2
    trees_3 = [[[0, 1, 1], [1, 0, 0], [1, 0, 0]]]
    assert list(nx.nonisomorphic_trees(3, create='matrix')) == trees_3
    trees_4 = [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]]
    assert list(nx.nonisomorphic_trees(4, create='matrix')) == trees_4