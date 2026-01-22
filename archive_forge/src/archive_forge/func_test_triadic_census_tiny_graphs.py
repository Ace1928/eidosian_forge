import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_tiny_graphs():
    tc = nx.triadic_census(nx.empty_graph(0, create_using=nx.DiGraph))
    assert {} == {typ: cnt for typ, cnt in tc.items() if cnt > 0}
    tc = nx.triadic_census(nx.empty_graph(1, create_using=nx.DiGraph))
    assert {} == {typ: cnt for typ, cnt in tc.items() if cnt > 0}
    tc = nx.triadic_census(nx.empty_graph(2, create_using=nx.DiGraph))
    assert {} == {typ: cnt for typ, cnt in tc.items() if cnt > 0}
    tc = nx.triadic_census(nx.DiGraph([(1, 2)]))
    assert {} == {typ: cnt for typ, cnt in tc.items() if cnt > 0}