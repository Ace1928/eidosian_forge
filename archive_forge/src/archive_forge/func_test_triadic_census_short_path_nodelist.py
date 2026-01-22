import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_short_path_nodelist():
    G = nx.path_graph('abc', create_using=nx.DiGraph)
    expected = {'021C': 1}
    for nl in ['a', 'b', 'c', 'ab', 'ac', 'bc', 'abc']:
        triad_census = nx.triadic_census(G, nodelist=nl)
        assert expected == {typ: cnt for typ, cnt in triad_census.items() if cnt > 0}