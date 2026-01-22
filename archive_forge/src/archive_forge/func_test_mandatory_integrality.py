from itertools import product
import pytest
import networkx as nx
def test_mandatory_integrality():
    byte_block_size = 32
    ex_1_broken = nx.DiGraph()
    ex_1_broken.add_edge(1, 2, **{EWL: 3.2})
    ex_1_broken.add_edge(1, 4, **{EWL: 2.4})
    ex_1_broken.add_edge(2, 3, **{EWL: 4.0})
    ex_1_broken.add_edge(2, 5, **{EWL: 6.3})
    ex_1_broken.nodes[1][NWL] = 1.2
    ex_1_broken.nodes[2][NWL] = 1
    ex_1_broken.nodes[3][NWL] = 1
    ex_1_broken.nodes[4][NWL] = 1
    ex_1_broken.nodes[5][NWL] = 2
    with pytest.raises(TypeError):
        nx.community.lukes_partitioning(ex_1_broken, byte_block_size, node_weight=NWL, edge_weight=EWL)