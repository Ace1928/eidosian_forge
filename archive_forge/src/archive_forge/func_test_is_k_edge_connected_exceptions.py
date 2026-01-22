import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_is_k_edge_connected_exceptions():
    pytest.raises(nx.NetworkXNotImplemented, is_locally_k_edge_connected, nx.DiGraph(), 1, 2, k=0)
    pytest.raises(nx.NetworkXNotImplemented, is_locally_k_edge_connected, nx.MultiGraph(), 1, 2, k=0)
    pytest.raises(ValueError, is_locally_k_edge_connected, nx.Graph(), 1, 2, k=0)