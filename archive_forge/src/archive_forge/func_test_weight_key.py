import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_weight_key():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.add_edges_from([(3, 8), (1, 2), (2, 3)])
    impossible = {(3, 6), (3, 9)}
    rng = random.Random(0)
    avail_uv = list(set(complement_edges(G)) - impossible)
    avail = [(u, v, {'cost': rng.random()}) for u, v in avail_uv]
    _augment_and_check(G, k=1)
    _augment_and_check(G, k=1, avail=avail_uv)
    _augment_and_check(G, k=1, avail=avail, weight='cost')
    _check_augmentations(G, avail, weight='cost')