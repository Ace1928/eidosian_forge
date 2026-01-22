import itertools
import os
import warnings
import pytest
import networkx as nx
def test_np_edgelist():
    nx.draw_networkx(barbell, edgelist=np.array([(0, 2), (0, 3)]))