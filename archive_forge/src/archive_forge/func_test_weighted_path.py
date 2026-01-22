import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_weighted_path():
    G = nx.path_graph(6)
    for i in range(5):
        G[i][i + 1]['w'] = i + 1
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, weight='w', relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    assert M[0][1]['weight'] == 2
    assert M[1][2]['weight'] == 4
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1