import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_metric(self):
    """Tests for providing an alternate distance metric to the generator."""
    G = nx.waxman_graph(50, 0.5, 0.1, metric=l1dist)
    assert len(G) == 50