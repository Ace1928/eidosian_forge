import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_bridge():
    G = nx.Graph([(2393, 2257), (2393, 2685), (2685, 2257), (1758, 2257)])
    _check_augmentations(G)