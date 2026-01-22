import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
@pytest.mark.parametrize('simrank_similarity', simrank_algs)
def test_simrank_source_no_target(self, simrank_similarity):
    G = nx.cycle_graph(5)
    expected = {0: 1, 1: 0.3951219505902448, 2: 0.5707317069281646, 3: 0.5707317069281646, 4: 0.3951219505902449}
    actual = simrank_similarity(G, source=0)
    assert expected == pytest.approx(actual, abs=0.01)
    G = nx.DiGraph()
    G.add_node(0, label='Univ')
    G.add_node(1, label='ProfA')
    G.add_node(2, label='ProfB')
    G.add_node(3, label='StudentA')
    G.add_node(4, label='StudentB')
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (4, 2), (3, 0)])
    expected = {0: 1, 1: 0.0, 2: 0.1323363991265798, 3: 0.0, 4: 0.03387811817640443}
    actual = simrank_similarity(G, importance_factor=0.8, source=0)
    assert expected == pytest.approx(actual, abs=0.01)