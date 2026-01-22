import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_davis_betweenness_centrality(self):
    G = self.davis
    bet = bipartite.betweenness_centrality(G, self.top_nodes)
    answer = {'E8': 0.24, 'E9': 0.23, 'E7': 0.13, 'Nora Fayette': 0.11, 'Evelyn Jefferson': 0.1, 'Theresa Anderson': 0.09, 'E6': 0.07, 'Sylvia Avondale': 0.07, 'Laura Mandeville': 0.05, 'Brenda Rogers': 0.05, 'Katherina Rogers': 0.05, 'E5': 0.04, 'Helen Lloyd': 0.04, 'E3': 0.02, 'Ruth DeSand': 0.02, 'Verne Sanderson': 0.02, 'E12': 0.02, 'Myra Liddel': 0.02, 'E11': 0.02, 'Eleanor Nye': 0.01, 'Frances Anderson': 0.01, 'Pearl Oglethorpe': 0.01, 'E4': 0.01, 'Charlotte McDowd': 0.01, 'E10': 0.01, 'Olivia Carleton': 0.01, 'Flora Price': 0.01, 'E2': 0.0, 'E1': 0.0, 'Dorothy Murchison': 0.0, 'E13': 0.0, 'E14': 0.0}
    for node, value in answer.items():
        assert value == pytest.approx(bet[node], abs=0.01)