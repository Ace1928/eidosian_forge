import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_networkx_arrowsize_incorrect_size():
    G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 3)])
    arrowsize = [1, 2, 3]
    with pytest.raises(ValueError, match='arrowsize should have the same length as edgelist'):
        nx.draw(G, arrowsize=arrowsize)