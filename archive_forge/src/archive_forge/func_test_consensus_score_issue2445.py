import numpy as np
from sklearn.metrics import consensus_score
from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.utils._testing import assert_almost_equal
def test_consensus_score_issue2445():
    """Different number of biclusters in A and B"""
    a_rows = np.array([[True, True, False, False], [False, False, True, True], [False, False, False, True]])
    a_cols = np.array([[True, True, False, False], [False, False, True, True], [False, False, False, True]])
    idx = [0, 2]
    s = consensus_score((a_rows, a_cols), (a_rows[idx], a_cols[idx]))
    assert_almost_equal(s, 2.0 / 3.0)