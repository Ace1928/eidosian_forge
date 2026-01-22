import pytest
import networkx as nx
from networkx import barbell_graph
from networkx.algorithms.community import modularity, partition_quality
from networkx.algorithms.community.quality import inter_community_edges
def test_modularity_resolution():
    G = nx.barbell_graph(3, 0)
    C = [{0, 1, 4}, {2, 3, 5}]
    assert modularity(G, C) == pytest.approx(3 / 7 - 100 / 14 ** 2)
    gamma = 2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(3 / 7 - gamma * 100 / 14 ** 2)
    gamma = 0.2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(3 / 7 - gamma * 100 / 14 ** 2)
    C = [{0, 1, 2}, {3, 4, 5}]
    assert modularity(G, C) == pytest.approx(6 / 7 - 98 / 14 ** 2)
    gamma = 2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(6 / 7 - gamma * 98 / 14 ** 2)
    gamma = 0.2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(6 / 7 - gamma * 98 / 14 ** 2)
    G = nx.barbell_graph(5, 3)
    C = [frozenset(range(5)), frozenset(range(8, 13)), frozenset(range(5, 8))]
    gamma = 1
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(22 / 24 - gamma * (918 / 48 ** 2))
    gamma = 2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(22 / 24 - gamma * (918 / 48 ** 2))
    gamma = 0.2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(22 / 24 - gamma * (918 / 48 ** 2))
    C = [{0, 1, 2, 3}, {9, 10, 11, 12}, {5, 6, 7}, {4}, {8}]
    gamma = 1
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(14 / 24 - gamma * (598 / 48 ** 2))
    gamma = 2.5
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(14 / 24 - gamma * (598 / 48 ** 2))
    gamma = 0.2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(14 / 24 - gamma * (598 / 48 ** 2))
    C = [frozenset(range(8)), frozenset(range(8, 13))]
    gamma = 1
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(23 / 24 - gamma * (1170 / 48 ** 2))
    gamma = 2
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(23 / 24 - gamma * (1170 / 48 ** 2))
    gamma = 0.3
    result = modularity(G, C, resolution=gamma)
    assert result == pytest.approx(23 / 24 - gamma * (1170 / 48 ** 2))