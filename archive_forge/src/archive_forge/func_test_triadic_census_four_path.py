import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_four_path():
    G = nx.path_graph('abcd', create_using=nx.DiGraph)
    expected = {'012': 2, '021C': 2}
    triad_census = nx.triadic_census(G)
    assert expected == {typ: cnt for typ, cnt in triad_census.items() if cnt > 0}