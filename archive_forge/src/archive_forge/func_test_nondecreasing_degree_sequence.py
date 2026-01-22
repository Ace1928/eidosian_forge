from itertools import groupby
import pytest
import networkx as nx
from networkx import graph_atlas, graph_atlas_g
from networkx.generators.atlas import NUM_GRAPHS
from networkx.utils import edges_equal, nodes_equal, pairwise
def test_nondecreasing_degree_sequence(self):
    exceptions = [('G55', 'G56'), ('G1007', 'G1008'), ('G1012', 'G1013')]
    for n, group in groupby(self.GAG, key=nx.number_of_nodes):
        for m, group in groupby(group, key=nx.number_of_edges):
            for G1, G2 in pairwise(group):
                if (G1.name, G2.name) in exceptions:
                    continue
                d1 = sorted((d for v, d in G1.degree()))
                d2 = sorted((d for v, d in G2.degree()))
                assert d1 <= d2