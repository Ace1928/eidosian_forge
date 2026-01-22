import math
from functools import partial
import pytest
import networkx as nx
def test_no_inter_cluster_common_neighbor(self):
    G = nx.complete_graph(4)
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 0
    G.nodes[2]['community'] = 0
    G.nodes[3]['community'] = 0
    self.test(G, [(0, 3)], [(0, 3, 2 / self.delta)])