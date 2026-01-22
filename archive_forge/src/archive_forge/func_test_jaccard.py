import numpy as np
from sklearn.metrics import consensus_score
from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.utils._testing import assert_almost_equal
def test_jaccard():
    a1 = np.array([True, True, False, False])
    a2 = np.array([True, True, True, True])
    a3 = np.array([False, True, True, False])
    a4 = np.array([False, False, True, True])
    assert _jaccard(a1, a1, a1, a1) == 1
    assert _jaccard(a1, a1, a2, a2) == 0.25
    assert _jaccard(a1, a1, a3, a3) == 1.0 / 7
    assert _jaccard(a1, a1, a4, a4) == 0