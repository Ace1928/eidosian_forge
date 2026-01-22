from itertools import groupby
import pytest
import networkx as nx
from networkx import graph_atlas, graph_atlas_g
from networkx.generators.atlas import NUM_GRAPHS
from networkx.utils import edges_equal, nodes_equal, pairwise
def test_nondecreasing_edges(self):
    for n, group in groupby(self.GAG, key=nx.number_of_nodes):
        for m1, m2 in pairwise(map(nx.number_of_edges, group)):
            assert m2 <= m1 + 1