from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_dimensionality(self):
    ntrial = 10
    for seed in range(1234, 1234 + ntrial):
        rg = nx.erdos_renyi_graph(10, 0.3, seed=seed)
        nnodes = rg.number_of_nodes()
        nedges = rg.number_of_edges()
        ncomp = nx.number_connected_components(rg)
        mcb = nx.minimum_cycle_basis(rg)
        assert len(mcb) == nedges - nnodes + ncomp
        check_independent(mcb)