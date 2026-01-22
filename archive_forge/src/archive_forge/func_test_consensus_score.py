import numpy as np
from sklearn.metrics import consensus_score
from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.utils._testing import assert_almost_equal
def test_consensus_score():
    a = [[True, True, False, False], [False, False, True, True]]
    b = a[::-1]
    assert consensus_score((a, a), (a, a)) == 1
    assert consensus_score((a, a), (b, b)) == 1
    assert consensus_score((a, b), (a, b)) == 1
    assert consensus_score((a, b), (b, a)) == 1
    assert consensus_score((a, a), (b, a)) == 0
    assert consensus_score((a, a), (a, b)) == 0
    assert consensus_score((b, b), (a, b)) == 0
    assert consensus_score((b, b), (b, a)) == 0