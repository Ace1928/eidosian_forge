import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_weights_thresholds(self):
    wseq = [3, 4, 3, 3, 5, 6, 5, 4, 5, 6]
    cs = nxt.weights_to_creation_sequence(wseq, threshold=10)
    wseq = nxt.creation_sequence_to_weights(cs)
    cs2 = nxt.weights_to_creation_sequence(wseq)
    assert cs == cs2
    wseq = nxt.creation_sequence_to_weights(nxt.uncompact([3, 1, 2, 3, 3, 2, 3]))
    assert wseq == [s * 0.125 for s in [4, 4, 4, 3, 5, 5, 2, 2, 2, 6, 6, 6, 1, 1, 7, 7, 7]]
    wseq = nxt.creation_sequence_to_weights([3, 1, 2, 3, 3, 2, 3])
    assert wseq == [s * 0.125 for s in [4, 4, 4, 3, 5, 5, 2, 2, 2, 6, 6, 6, 1, 1, 7, 7, 7]]
    wseq = nxt.creation_sequence_to_weights(list(enumerate('ddidiiidididi')))
    assert wseq == [s * 0.1 for s in [5, 5, 4, 6, 3, 3, 3, 7, 2, 8, 1, 9, 0]]
    wseq = nxt.creation_sequence_to_weights('ddidiiidididi')
    assert wseq == [s * 0.1 for s in [5, 5, 4, 6, 3, 3, 3, 7, 2, 8, 1, 9, 0]]
    wseq = nxt.creation_sequence_to_weights('ddidiiidididid')
    ws = [s / 12 for s in [6, 6, 5, 7, 4, 4, 4, 8, 3, 9, 2, 10, 1, 11]]
    assert sum((abs(c - d) for c, d in zip(wseq, ws))) < 1e-14