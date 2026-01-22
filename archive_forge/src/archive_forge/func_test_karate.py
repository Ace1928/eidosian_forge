import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_karate():
    G = nx.karate_club_graph()
    _check_separating_sets(G)