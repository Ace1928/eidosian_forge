import pytest
import networkx as nx
from networkx import barbell_graph
from networkx.algorithms.community import modularity, partition_quality
from networkx.algorithms.community.quality import inter_community_edges
def test_bad_partition(self):
    """Tests that a poor partition has a low coverage measure."""
    G = barbell_graph(3, 0)
    partition = [{0, 1, 4}, {2, 3, 5}]
    assert 3 / 7 == pytest.approx(partition_quality(G, partition)[0], abs=1e-07)