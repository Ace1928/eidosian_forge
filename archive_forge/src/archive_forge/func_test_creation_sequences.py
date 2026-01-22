import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_creation_sequences(self):
    deg = [3, 2, 2, 1]
    G = nx.generators.havel_hakimi_graph(deg)
    with pytest.raises(ValueError):
        nxt.creation_sequence(deg, with_labels=True, compact=True)
    cs0 = nxt.creation_sequence(deg)
    H0 = nxt.threshold_graph(cs0)
    assert ''.join(cs0) == 'ddid'
    cs1 = nxt.creation_sequence(deg, with_labels=True)
    H1 = nxt.threshold_graph(cs1)
    assert cs1 == [(1, 'd'), (2, 'd'), (3, 'i'), (0, 'd')]
    cs2 = nxt.creation_sequence(deg, compact=True)
    H2 = nxt.threshold_graph(cs2)
    assert cs2 == [2, 1, 1]
    assert ''.join(nxt.uncompact(cs2)) == 'ddid'
    assert graph_could_be_isomorphic(H0, G)
    assert graph_could_be_isomorphic(H0, H1)
    assert graph_could_be_isomorphic(H0, H2)