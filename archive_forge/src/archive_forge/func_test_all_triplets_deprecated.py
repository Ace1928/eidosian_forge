import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_all_triplets_deprecated():
    G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    with pytest.deprecated_call():
        nx.all_triplets(G)