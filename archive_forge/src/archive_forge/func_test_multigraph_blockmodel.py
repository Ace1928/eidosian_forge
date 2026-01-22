import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_multigraph_blockmodel():
    G = nx.MultiGraph(nx.path_graph(6))
    partition = [[0, 1], [2, 3], [4, 5]]
    M = nx.quotient_graph(G, partition, create_using=nx.MultiGraph(), relabel=True)
    assert nodes_equal(M.nodes(), [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M.nodes():
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1.0