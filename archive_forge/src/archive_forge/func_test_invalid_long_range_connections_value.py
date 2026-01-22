import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_invalid_long_range_connections_value(self):
    with pytest.raises(nx.NetworkXException, match='.*q must be >= 0'):
        nx.navigable_small_world_graph(5, p=1, q=-1, dim=1)