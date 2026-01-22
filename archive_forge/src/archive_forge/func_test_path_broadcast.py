import math
import networkx as nx
def test_path_broadcast():
    for i in range(2, 12):
        G = nx.path_graph(i)
        b_T, b_C = nx.tree_broadcast_center(G)
        assert b_T == math.ceil(i / 2)
        assert b_C == {math.ceil(i / 2), math.floor(i / 2), math.ceil(i / 2 - 1), math.floor(i / 2 - 1)}
        assert nx.tree_broadcast_time(G) == i - 1