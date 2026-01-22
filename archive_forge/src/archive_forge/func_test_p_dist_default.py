import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_p_dist_default(self):
    """Tests default p_dict = 0.5 returns graph with edge count <= RGG with
        same n, radius, dim and positions
        """
    nodes = 50
    dim = 2
    pos = {v: [random.random() for i in range(dim)] for v in range(nodes)}
    RGG = nx.random_geometric_graph(50, 0.25, pos=pos)
    SRGG = nx.soft_random_geometric_graph(50, 0.25, pos=pos)
    assert len(SRGG.edges()) <= len(RGG.edges())