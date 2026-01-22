import networkx as nx
def test_subtree_union():
    uf = nx.utils.UnionFind()
    uf.union(1, 2)
    uf.union(3, 4)
    uf.union(4, 5)
    uf.union(1, 5)
    assert list(uf.to_sets()) == [{1, 2, 3, 4, 5}]