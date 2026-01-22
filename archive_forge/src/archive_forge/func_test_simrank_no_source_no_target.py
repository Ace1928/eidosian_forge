import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
@pytest.mark.parametrize('simrank_similarity', simrank_algs)
def test_simrank_no_source_no_target(self, simrank_similarity):
    G = nx.cycle_graph(5)
    expected = {0: {0: 1, 1: 0.3951219505902448, 2: 0.5707317069281646, 3: 0.5707317069281646, 4: 0.3951219505902449}, 1: {0: 0.3951219505902448, 1: 1, 2: 0.3951219505902449, 3: 0.5707317069281646, 4: 0.5707317069281646}, 2: {0: 0.5707317069281646, 1: 0.3951219505902449, 2: 1, 3: 0.3951219505902449, 4: 0.5707317069281646}, 3: {0: 0.5707317069281646, 1: 0.5707317069281646, 2: 0.3951219505902449, 3: 1, 4: 0.3951219505902449}, 4: {0: 0.3951219505902449, 1: 0.5707317069281646, 2: 0.5707317069281646, 3: 0.3951219505902449, 4: 1}}
    actual = simrank_similarity(G)
    for k, v in expected.items():
        assert v == pytest.approx(actual[k], abs=0.01)
    G = nx.DiGraph()
    G.add_node(0, label='Univ')
    G.add_node(1, label='ProfA')
    G.add_node(2, label='ProfB')
    G.add_node(3, label='StudentA')
    G.add_node(4, label='StudentB')
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (4, 2), (3, 0)])
    expected = {0: {0: 1, 1: 0.0, 2: 0.1323363991265798, 3: 0.0, 4: 0.03387811817640443}, 1: {0: 0.0, 1: 1, 2: 0.4135512472705618, 3: 0.0, 4: 0.10586911930126384}, 2: {0: 0.1323363991265798, 1: 0.4135512472705618, 2: 1, 3: 0.04234764772050554, 4: 0.08822426608438655}, 3: {0: 0.0, 1: 0.0, 2: 0.04234764772050554, 3: 1, 4: 0.3308409978164495}, 4: {0: 0.03387811817640443, 1: 0.10586911930126384, 2: 0.08822426608438655, 3: 0.3308409978164495, 4: 1}}
    actual = simrank_similarity(G, importance_factor=0.8)
    for k, v in expected.items():
        assert v == pytest.approx(actual[k], abs=0.01)