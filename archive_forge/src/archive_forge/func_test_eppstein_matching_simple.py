import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_eppstein_matching_simple(self):
    match = eppstein_matching(self.simple_graph)
    assert match == self.simple_solution