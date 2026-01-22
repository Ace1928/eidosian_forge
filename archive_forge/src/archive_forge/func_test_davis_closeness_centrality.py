import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_davis_closeness_centrality(self):
    G = self.davis
    clos = bipartite.closeness_centrality(G, self.top_nodes)
    answer = {'E8': 0.85, 'E9': 0.79, 'E7': 0.73, 'Nora Fayette': 0.8, 'Evelyn Jefferson': 0.8, 'Theresa Anderson': 0.8, 'E6': 0.69, 'Sylvia Avondale': 0.77, 'Laura Mandeville': 0.73, 'Brenda Rogers': 0.73, 'Katherina Rogers': 0.73, 'E5': 0.59, 'Helen Lloyd': 0.73, 'E3': 0.56, 'Ruth DeSand': 0.71, 'Verne Sanderson': 0.71, 'E12': 0.56, 'Myra Liddel': 0.69, 'E11': 0.54, 'Eleanor Nye': 0.67, 'Frances Anderson': 0.67, 'Pearl Oglethorpe': 0.67, 'E4': 0.54, 'Charlotte McDowd': 0.6, 'E10': 0.55, 'Olivia Carleton': 0.59, 'Flora Price': 0.59, 'E2': 0.52, 'E1': 0.52, 'Dorothy Murchison': 0.65, 'E13': 0.52, 'E14': 0.52}
    for node, value in answer.items():
        assert value == pytest.approx(clos[node], abs=0.01)