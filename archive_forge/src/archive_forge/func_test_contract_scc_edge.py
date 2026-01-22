import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_contract_scc_edge(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 3)
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    assert sorted(cG.nodes()) == [0, 1]
    if 1 in scc[0]:
        edge = (0, 1)
    else:
        edge = (1, 0)
    assert list(cG.edges()) == [edge]