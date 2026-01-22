import pytest
import networkx
import networkx as nx
from .historical_tests import HistoricalTests
def test_reverse3(self):
    H = nx.DiGraph()
    H.add_nodes_from([1, 2, 3, 4])
    HR = H.reverse()
    assert sorted(HR.nodes()) == [1, 2, 3, 4]