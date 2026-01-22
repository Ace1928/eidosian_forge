import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_contract_scc1(self):
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (2, 11), (2, 12), (3, 4), (4, 3), (4, 5), (5, 6), (6, 5), (6, 7), (7, 8), (7, 9), (7, 10), (8, 9), (9, 7), (10, 6), (11, 2), (11, 4), (11, 6), (12, 6), (12, 11)])
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    assert nx.is_directed_acyclic_graph(cG)
    assert sorted(cG.nodes()) == [0, 1, 2, 3]
    mapping = {}
    for i, component in enumerate(scc):
        for n in component:
            mapping[n] = i
    edge = (mapping[2], mapping[3])
    assert cG.has_edge(*edge)
    edge = (mapping[2], mapping[5])
    assert cG.has_edge(*edge)
    edge = (mapping[3], mapping[5])
    assert cG.has_edge(*edge)