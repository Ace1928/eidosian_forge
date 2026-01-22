import pytest
from networkx import triad_graph
def test_triad_graph():
    G = triad_graph('030T')
    assert [tuple(e) for e in ('ab', 'ac', 'cb')] == sorted(G.edges())