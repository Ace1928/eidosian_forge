import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
def test_karate_0():
    G = nx.karate_club_graph()
    _check_connectivity(G)