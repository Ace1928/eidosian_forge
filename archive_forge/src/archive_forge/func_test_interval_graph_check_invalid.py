import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_check_invalid(self):
    """Tests for conditions that raise Exceptions"""
    invalids_having_none = [None, (1, 2)]
    with pytest.raises(TypeError):
        interval_graph(invalids_having_none)
    invalids_having_set = [{1, 2}]
    with pytest.raises(TypeError):
        interval_graph(invalids_having_set)
    invalids_having_seq_but_not_length2 = [(1, 2, 3)]
    with pytest.raises(TypeError):
        interval_graph(invalids_having_seq_but_not_length2)
    invalids_interval = [[3, 2]]
    with pytest.raises(ValueError):
        interval_graph(invalids_interval)