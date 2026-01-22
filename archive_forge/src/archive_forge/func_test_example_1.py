import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_example_1():
    G = graph_example_1()
    _check_separating_sets(G)